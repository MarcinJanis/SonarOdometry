import torch
import torch.nn.functional as F
import torch.nn as nn 
from kornia.feature import LoFTR

import numpy as np 
import cv2 

class sonar_odometry(nn.Module):

    def __init__(self, model_config, sonar_config, device, 
                 depth_compesation=True,
                 key_frames=True,
                 input_img_format='polar',
                 ref_frame_orient='sim', # 'sim', 'aracati'
                 use_fls_filter=True # Dodana flaga dla pre-processingu CLAHE
                 ):
        
        super().__init__()
        self.device = device 

        # extrinsics calibration
        yaw_offset = sonar_config.position.yaw
        x_offset = sonar_config.position.x
        y_offset = sonar_config.position.y
        
        # Trnasform Robot -> Sonar
        self.T_R_S_2d = np.array([
            [np.cos(yaw_offset), -np.sin(yaw_offset), x_offset],
            [np.sin(yaw_offset),  np.cos(yaw_offset), y_offset],
            [0,                   0,                  1]
        ])
        # Transform Sonar -> Robot 
        self.T_S_R_2d = np.linalg.inv(self.T_R_S_2d)
        
        # System configuration
        self.ref_frame_orient = ref_frame_orient 
        self.depth_compesation = depth_compesation
        self.key_frames = key_frames 
        self.use_fls_filter = getattr(model_config, 'use_fls_filter', use_fls_filter)
        
        # Parameters
        self.key_frames_min_dist = model_config.key_frames_min_dist
        self.key_frames_min_rot = model_config.key_frames_min_rot

        self.inliers_low_threshold = model_config.inliers_low_threshold
        self.pts_match_thresh = model_config.pts_match_thresh
        self.ransac_thresh = model_config.ransac_thresh
        
        # Gating i Recovery
        self.max_trans_step = getattr(model_config, 'max_trans_step', 0.5)
        self.max_rot_step = getattr(model_config, 'max_rot_step', 0.20)
        self.min_inliers_abs = getattr(model_config, 'min_inliers_abs', 15)
        self.min_inliers_ratio = getattr(model_config, 'min_inliers_ratio', 0.07)
        self.max_skip_frames = getattr(model_config, 'max_skip_frames', 3)

        # CVM State (Model Stałej Prędkości)
        self.last_step_tx, self.last_step_ty, self.last_step_theta = 0.0, 0.0, 0.0
        self.blind_frames = 0
        self.cvm_decay = 0.85 # Współczynnik hamowania w przypadku utraty cech

        # Sonar congifuration
        self.input_img_format = input_img_format
        if self.input_img_format == 'polar':
            self.cart_frame_size = (model_config.POLAR_FLS_INPUT_HEIGHT, 2 * model_config.POLAR_FLS_INPUT_HEIGHT)
        else:
            self.cart_frame_size = (model_config.CART_FLS_INPUT_HEIGHT, model_config.CART_FLS_INPUT_WIDTH)
    
        self.r_min = sonar_config.range.min
        self.r_max = sonar_config.range.max
        self.theta_max = sonar_config.fov.horizontal

        self.match_points = LoFTR(pretrained='outdoor').to(device).eval()
        self.window_size = 3 
        self.sliding_window = [] 
        
        self.current_pose = None
        self.last_frame_data = None # Cache dla klatki t-1 (Fallback kaskadowy)
        
        self.polar2cart_grid = None
        self.polar2cart_mask = None

    def fls_filter(self, frame):
        """ Filtracja obrazu: MedianBlur + Bilateral + CLAHE (Zapożyczone z zeszytu) """
        device = frame.device 
        b, c, h, w = frame.shape
        out_frames = []
        for i in range(b):
            frame_np = frame[i, 0].detach().cpu().numpy()
            frame_uint8 = np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)
            
            # 1. Zmiękczanie szumu plamkowego
            blured = cv2.medianBlur(frame_uint8, ksize=3)
            # 2. Filtr bilateralny z zachowaniem mocnych ech 
            bilateral = cv2.bilateralFilter(blured, d=5, sigmaColor=25, sigmaSpace=5)
            # 3. Wyrównanie kontrastu
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            filtered = clahe.apply(bilateral)
            
            filtered_float = filtered.astype(np.float32) / 255.0
            out_frames.append(torch.tensor(filtered_float, device=device).unsqueeze(0))
        return torch.stack(out_frames, dim=0)

    def set_init_state(self, init_x, init_y, init_azimuth, init_frame, carth_mask=None):
        if self.use_fls_filter:
            init_frame = self.fls_filter(init_frame)

        b, c, h, w = init_frame.shape
        out_h, out_w = h, 2 * h

        self.cart_frame_size = (out_h, out_w)

        y = torch.arange(out_h, device=self.device, dtype=torch.float32)
        x = torch.arange(out_w, device=self.device, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing='ij')
        x = x - out_w / 2.0
        y = out_h - y

        scale = (self.r_max - self.r_min) / out_h
        x_r = x * scale
        y_r = y * scale + self.r_min
        r = torch.sqrt(x_r**2 + y_r**2)
        theta = torch.atan2(x_r, torch.clamp(y_r, min=1e-5))
        
        norm_theta = theta / (self.theta_max / 2.0)
        norm_r = (r - self.r_min) / (self.r_max - self.r_min) * 2.0 - 1.0

        self.polar2cart_grid = torch.stack((norm_theta, -norm_r), dim=-1).unsqueeze(0) 
        
        if self.input_img_format == 'polar':
            valid_mask = (norm_theta >= -1.0) & (norm_theta <= 1.0) & (norm_r >= -1.0) & (norm_r <= 1.0)
            self.polar2cart_mask = valid_mask.unsqueeze(0).expand(b, -1, -1).float()
        elif carth_mask is not None:
            self.polar2cart_mask = carth_mask 
        else:
            init_frame_np = init_frame.view(h, w).detach().cpu().numpy()
            mask = (init_frame_np == 0.0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            self.polar2cart_mask = torch.tensor(cleaned_mask, device=init_frame.device, dtype=torch.float).unsqueeze(0)

        init_pose = np.array([[np.cos(init_azimuth), -np.sin(init_azimuth), init_x], 
                              [np.sin(init_azimuth),  np.cos(init_azimuth), init_y], 
                              [0,                     0,                    1]])
        
        first_frame = self.polar2car(init_frame)
        self.current_pose = init_pose
        self.sliding_window = [(first_frame, self.polar2cart_mask, init_pose)]
        
        # Kopia dla kaskadowego dopasowania (fallback)
        self.last_frame_data = (first_frame, self.polar2cart_mask, init_pose)
        
        self.blind_frames = 0
        self.last_step_tx, self.last_step_ty, self.last_step_theta = 0.0, 0.0, 0.0

    def _match_and_estimate(self, ref_frame, ref_mask, ref_pose, new_frame, depth):
        """ Helper method do wyliczania dopasowań i estymacji pozy """
        matches = self.match_points({'image0': ref_frame, 'mask0': ref_mask, 'image1': new_frame, 'mask1': self.polar2cart_mask})
        pts1, pts2, confidence = matches['keypoints0'], matches['keypoints1'], matches['confidence']
        valid_matches = confidence > self.pts_match_thresh
        pts1, pts2 = pts1[valid_matches], pts2[valid_matches]
        
        if len(pts1) < 3: return None 

        pts1_r, pts2_r = self.scale_px2physcial(pts1), self.scale_px2physcial(pts2)
        
        if self.depth_compesation:
            ray1, ray2 = torch.sqrt(pts1_r[:, 0]**2 + pts1_r[:, 1]**2), torch.sqrt(pts2_r[:, 0]**2 + pts2_r[:, 1]**2)
            valid_mask = (ray1 > depth) & (ray2 > depth)
            pts1_r = pts1_r[valid_mask] * ((torch.sqrt(ray1[valid_mask]**2 - depth**2) / ray1[valid_mask]).unsqueeze(1))
            pts2_r = pts2_r[valid_mask] * ((torch.sqrt(ray2[valid_mask]**2 - depth**2) / ray2[valid_mask]).unsqueeze(1))

        pts1_np, pts2_np = pts1_r.cpu().numpy(), pts2_r.cpu().numpy()
        if len(pts1_np) < 3: return None

        M, inlier_mask = cv2.estimateAffinePartial2D(pts2_np, pts1_np, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh, maxIters=3000, confidence=0.999)

        if M is not None and inlier_mask is not None:
            inlier_mask = inlier_mask.ravel().astype(bool)
            inliers_abs = int(inlier_mask.sum())
            
            if inliers_abs >= self.min_inliers_abs:
                angle = np.arctan2(M[1, 0], M[0, 0])
                R_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                diffs = pts1_np[inlier_mask] - (R_mat @ pts2_np[inlier_mask].T).T
                
                raw_tx_sonar, raw_ty_sonar = (float(np.median(diffs[:, 0])), float(np.median(diffs[:, 1])))
                
                theta = -angle if self.ref_frame_orient == 'sim' else angle
                tx, ty = (raw_ty_sonar, -raw_tx_sonar) if self.ref_frame_orient == 'sim' else (-raw_ty_sonar, -raw_tx_sonar)
                
                local_T = np.array([[np.cos(theta), -np.sin(theta), tx], [np.sin(theta), np.cos(theta), ty], [0, 0, 1]])
                est_pose = ref_pose @ (self.T_R_S_2d @ local_T @ self.T_S_R_2d)
                
                return {
                    'est_pose': est_pose,
                    'pts1': pts1, 
                    'pts2': pts2, 
                    'confidence': confidence,
                    'inliers_abs': inliers_abs, 
                    'inliers_p': inliers_abs / len(pts1) if len(pts1) > 0 else 0,
                    'ref_frame': ref_frame,
                    'raw_tx_sonar': raw_tx_sonar, 
                    'raw_ty_sonar': raw_ty_sonar
                }
        return None

    @torch.no_grad()
    def forward(self, frame, depth, return_visu=False):
        if self.use_fls_filter:
            frame = self.fls_filter(frame)

        new_frame = self.polar2car(frame) if self.input_img_format == 'polar' else frame
        
        est_poses = []
        latest_visu_match = None 

        # 1. SLIDING WINDOW MATCHING (Keyframes)
        for i, (ref_frame, ref_mask, ref_pose) in enumerate(self.sliding_window):
            match_res = self._match_and_estimate(ref_frame, ref_mask, ref_pose, new_frame, depth)
            if match_res is not None:
                est_poses.append(match_res)
                if i == len(self.sliding_window) - 1:
                    latest_visu_match = match_res

        # 2. CASCADE FALLBACK (Kaskadowe Dopasowywanie do klatki t-1)
        if len(est_poses) == 0 and self.last_frame_data is not None:
            ref_frame, ref_mask, ref_pose = self.last_frame_data
            match_res = self._match_and_estimate(ref_frame, ref_mask, ref_pose, new_frame, depth)
            if match_res is not None:
                est_poses.append(match_res)
                latest_visu_match = match_res

        # 3. GATING & CVM LOGIC
        use_cvm = False
        if len(est_poses) > 0:
            est_x_list = [p['est_pose'][0, 2] for p in est_poses]
            est_y_list = [p['est_pose'][1, 2] for p in est_poses]
            est_yaw_list = [np.arctan2(p['est_pose'][1, 0], p['est_pose'][0, 0]) for p in est_poses]
            
            median_x, median_y = np.median(est_x_list), np.median(est_y_list)
            median_azimuth = np.arctan2(np.sum(np.sin(est_yaw_list)), np.sum(np.cos(est_yaw_list)))
            
            raw_new_pose = np.array([[np.cos(median_azimuth), -np.sin(median_azimuth), median_x], 
                                     [np.sin(median_azimuth), np.cos(median_azimuth), median_y], 
                                     [0, 0, 1]])
            
            R_curr, t_curr = self.current_pose[0:2, 0:2], self.current_pose[0:2, 2]
            t_step = R_curr.T @ (raw_new_pose[0:2, 2] - t_curr)
            step_theta = float(np.arctan2(raw_new_pose[1, 0], raw_new_pose[0, 0]) - np.arctan2(R_curr[1, 0], R_curr[0, 0]))
            
            # Normalizacja kąta
            step_theta = np.arctan2(np.sin(step_theta), np.cos(step_theta))
            
            multiplier = min(self.blind_frames + 1, self.max_skip_frames + 1)
            
            is_valid = (abs(t_step[0]) < self.max_trans_step * multiplier) and \
                       (abs(t_step[1]) < self.max_trans_step * multiplier) and \
                       (abs(step_theta) < self.max_rot_step * multiplier)
            
            if is_valid:
                new_pose = raw_new_pose
                self.last_step_tx, self.last_step_ty, self.last_step_theta = float(t_step[0]), float(t_step[1]), float(step_theta)
                self.blind_frames = 0
            else:
                use_cvm = True
        else:
            use_cvm = True

        if use_cvm:
            self.blind_frames += 1
            
            # Damping/Tłumienie: Wygaszanie CVM z każdą kolejną ślepą klatką
            self.last_step_tx *= self.cvm_decay
            self.last_step_ty *= self.cvm_decay
            self.last_step_theta *= self.cvm_decay
            
            cvm_T = np.array([[np.cos(self.last_step_theta), -np.sin(self.last_step_theta), self.last_step_tx],
                              [np.sin(self.last_step_theta),  np.cos(self.last_step_theta), self.last_step_ty],
                              [0, 0, 1]])
            new_pose = self.current_pose @ cvm_T

        global_x, global_y = float(new_pose[0, 2]), float(new_pose[1, 2])
        global_azimuth = float(np.arctan2(new_pose[1, 0], new_pose[0, 0]))

        # 4. KEYFRAME LOGIC (Z wymuszonym Resetem)
        _, _, latest_kf_pose = self.sliding_window[-1]
        dist = np.sqrt((global_x - latest_kf_pose[0, 2])**2 + (global_y - latest_kf_pose[1, 2])**2)
        
        prev_azimuth = np.arctan2(latest_kf_pose[1, 0], latest_kf_pose[0, 0])
        azimuth_diff = np.abs(np.arctan2(np.sin(global_azimuth - prev_azimuth), np.cos(global_azimuth - prev_azimuth)))
        
        # Wymuszamy klatkę kluczową w przypadku timeout'u (blind_frames), 
        # aby zresetować bazę wizualną po utracie stabilności
        key_frame_detected = (dist >= self.key_frames_min_dist or 
                              azimuth_diff >= self.key_frames_min_rot or 
                              self.blind_frames >= self.max_skip_frames)
        
        if key_frame_detected:
            self.sliding_window.append((new_frame, self.polar2cart_mask, new_pose))
            if len(self.sliding_window) > self.window_size: 
                self.sliding_window.pop(0)
            
            # Jeśli wymusiliśmy reset z powodu ślepoty, ucinamy naliczanie
            if self.blind_frames >= self.max_skip_frames: 
                self.blind_frames = 0

        # Zawsze zapisujemy klatkę do Cache na potrzeby fallbacku
        self.last_frame_data = (new_frame, self.polar2cart_mask, new_pose)
        self.current_pose = new_pose

        if not return_visu:
            return (global_x, global_y), global_azimuth
        else:
            R_kf, t_kf = latest_kf_pose[0:2, 0:2], latest_kf_pose[0:2, 2]
            R_new, t_new = new_pose[0:2, 0:2], new_pose[0:2, 2]

            R_rel = R_kf.T @ R_new
            t_rel = R_kf.T @ (t_new - t_kf)

            theta_effective = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
            tx_effective = float(t_rel[0])
            ty_effective = float(t_rel[1])

            b, c, h, w = new_frame.shape
            if latest_visu_match is not None:
                frame1_np = latest_visu_match['ref_frame'].squeeze(0).permute(1, 2, 0).cpu().numpy()
                pts1_visu = latest_visu_match['pts1'].cpu().numpy()
                pts2_visu = latest_visu_match['pts2'].cpu().numpy()
                conf_visu = float(latest_visu_match['confidence'].mean().cpu().numpy())
                v_inliers_abs = latest_visu_match['inliers_abs']
                v_inliers_ratio = latest_visu_match['inliers_p']
                v_raw_tx_sonar = latest_visu_match['raw_tx_sonar']
                v_raw_ty_sonar = latest_visu_match['raw_ty_sonar']
            else:
                frame1_np = self.sliding_window[-1][0].squeeze(0).permute(1, 2, 0).cpu().numpy()
                pts1_visu, pts2_visu = np.zeros((0,2)), np.zeros((0,2))
                conf_visu, v_inliers_abs, v_inliers_ratio = 0.0, 0, 0.0
                v_raw_tx_sonar, v_raw_ty_sonar = 0.0, 0.0

            frame2_np = new_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            combined_img_gray = np.concatenate((frame1_np, frame2_np), axis=1)
            frames_np_rgb = cv2.cvtColor(combined_img_gray, cv2.COLOR_GRAY2RGB) if combined_img_gray.shape[-1] == 1 else combined_img_gray
            
            # Zapobieganie błędom przy pakowaniu do listy jeśli est_poses jest puste
            if len(est_poses) > 0:
                est_x_list = [p['est_pose'][0, 2] for p in est_poses]
                est_y_list = [p['est_pose'][1, 2] for p in est_poses]
                est_yaw_list = [np.arctan2(p['est_pose'][1, 0], p['est_pose'][0, 0]) for p in est_poses]
                individual_estimates = list(zip(est_x_list, est_y_list, est_yaw_list))
            else:
                individual_estimates = []

            visu = {
                'combined_imgs': frames_np_rgb,
                'pts1': pts1_visu,
                'pts2': pts2_visu,
                'pts2_offset': (0, w),
                'inliers_ratio': v_inliers_ratio,
                'inliers_abs': v_inliers_abs,
                'matches_total': len(pts1_visu),
                'mean_matched_confidence': conf_visu,
                'key_frame_detected': key_frame_detected,
                'step_is_valid': not use_cvm,
                'tx_sonar': float(v_raw_tx_sonar),
                'ty_sonar': float(v_raw_ty_sonar),
                'tx_mapped': tx_effective,
                'ty_mapped': ty_effective,
                'theta': theta_effective,
                'displacement': float(dist),
                'azimuth_diff': float(azimuth_diff),
                'global_pose': (global_x, global_y, global_azimuth),
                'skipped_frames': self.blind_frames,
                'window_matches_count': f"{len(est_poses)}/{len(self.sliding_window)}",
                'individual_estimates': individual_estimates
            }
            return (global_x, global_y), global_azimuth, visu

    def polar2car(self, frame):
        out = F.grid_sample(frame, self.polar2cart_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return out * self.polar2cart_mask.unsqueeze(1)

    def scale_px2physcial(self, pts_px):
        out_h, out_w = self.cart_frame_size
        scale = (self.r_max - self.r_min) / out_h
        x = (pts_px[:, 0] - out_w / 2.0) * scale
        y = (out_h - pts_px[:, 1]) * scale + self.r_min
        return torch.stack([x, y], dim=1)

# ===================================================
# This works: but it can be improved 

# import torch
# import torch.nn.functional as F
# import torch.nn as nn 
# from kornia.feature import LoFTR

# import numpy as np 
# import cv2 

# from box import Box
# import yaml

# from .utils import ExtrinsicsCalib

# class sonar_odometry(nn.Module):

#     def __init__(self, model_config, sonar_config, device, 
#                  depth_compesation=True,
#                  key_frames=True,
#                  input_img_format='polar',
#                  ref_frame_orient='sim' # 'sim', 'aracati'
#                  ):
        
#         super().__init__()

#         self.device = device 

#         self.calib = ExtrinsicsCalib(
#             T=[sonar_config.position.x, sonar_config.position.y, sonar_config.position.z],
#             R=[sonar_config.position.roll, sonar_config.position.pitch, sonar_config.position.yaw]
#         )

#         # --- 2D Lever-Arm Calibration Matrices (Sonar <-> Robot Base) ---
#         yaw_offset = sonar_config.position.yaw
#         x_offset = sonar_config.position.x
#         y_offset = sonar_config.position.y
        
#         # Matrix transforming a point in Sonar Frame to Robot Frame (T_Robot_Sonar)
#         self.T_R_S_2d = np.array([
#             [np.cos(yaw_offset), -np.sin(yaw_offset), x_offset],
#             [np.sin(yaw_offset),  np.cos(yaw_offset), y_offset],
#             [0,                   0,                  1]
#         ])
#         # Matrix transforming a point in Robot Frame to Sonar Frame (T_Sonar_Robot)
#         self.T_S_R_2d = np.linalg.inv(self.T_R_S_2d)
#         # ----------------------------------------------------------------

#         # --- init parameters --- 
#         self.ref_frame_orient = ref_frame_orient
#         self.depth_compesation = depth_compesation

#         self.key_frames = key_frames
#         self.key_frames_min_dist = model_config.key_frames_min_dist # [m]
#         self.key_frames_min_rot = model_config.key_frames_min_rot # [rad]
#         self.inliers_low_threshold = model_config.inliers_low_threshold

#         self.pts_match_thresh = model_config.pts_match_thresh # [-]
#         self.ransac_thresh = model_config.ransac_thresh # [m]

#         # --- NEW GATING & RECOVERY PARAMETERS ---
#         # Ustawiamy domyślne, bezpieczne wartości na wypadek ich braku w pliku YAML
#         self.max_trans_step = getattr(model_config, 'max_trans_step', 0.5)
#         self.max_rot_step = getattr(model_config, 'max_rot_step', 0.20)
#         self.min_inliers_abs = getattr(model_config, 'min_inliers_abs', 15)
#         self.min_inliers_ratio = getattr(model_config, 'min_inliers_ratio', 0.07)
#         self.max_skip_frames = getattr(model_config, 'max_skip_frames', 3)

#         self.input_img_format = input_img_format
#         if self.input_img_format == 'polar':
#             self.cart_frame_size = (model_config.POLAR_FLS_INPUT_HEIGHT, 2 * model_config.POLAR_FLS_INPUT_HEIGHT)
#         else:
#             self.cart_frame_size = (model_config.CART_FLS_INPUT_HEIGHT, model_config.CART_FLS_INPUT_WIDTH)
    
#         self.r_min = sonar_config.range.min
#         self.r_max = sonar_config.range.max
#         self.theta_max = sonar_config.fov.horizontal

#         # --- init modules ---
#         pretrained = 'outdoor'
#         self.match_points = LoFTR(pretrained=pretrained).to(device).eval()

#         # --- MULTI-REFERENCE CONSENSUS WINDOW STATE --- 
#         self.window_size = 3 
#         self.sliding_window = [] # Tuples: (frame_tensor, mask_tensor, global_pose_matrix)
#         self.current_pose = None
#         self.skip_frames = 1
        
#         self.polar2cart_grid = None
#         self.polar2cart_mask = None


#     def set_init_state(self, init_x, init_y, init_azimuth, init_frame, carth_mask=None):
#         b, c, h, w = init_frame.shape
#         out_h, out_w = h, 2 * h

#         # Dynamiczne nadpisanie rozmiaru konfiguracyjnego rzeczywistymi wymiarami
#         self.cart_frame_size = (out_h, out_w)

#         # Inverse remapping grid generation
#         y = torch.arange(out_h, device=self.device, dtype=torch.float32)
#         x = torch.arange(out_w, device=self.device, dtype=torch.float32)
#         y, x = torch.meshgrid(y, x, indexing='ij')

#         x = x - out_w / 2.0
#         y = out_h - y

#         scale = (self.r_max - self.r_min) / out_h
#         x_r = x * scale
#         y_r = y * scale + self.r_min

#         r = torch.sqrt(x_r**2 + y_r**2)
#         y_r_clamp = torch.clamp(y_r, min=1e-5)
#         theta = torch.atan2(x_r, y_r_clamp)

#         norm_theta = theta / (self.theta_max / 2.0)
#         norm_r = (r - self.r_min) / (self.r_max - self.r_min) * 2.0 - 1.0

#         self.polar2cart_grid = torch.stack((norm_theta, -norm_r), dim=-1).unsqueeze(0) 

#         if self.input_img_format == 'polar':
#             valid_mask = (norm_theta >= -1.0) & (norm_theta <= 1.0) & (norm_r >= -1.0) & (norm_r <= 1.0)
#             self.polar2cart_mask = valid_mask.unsqueeze(0).expand(b, -1, -1).float()
#         elif carth_mask is not None:
#             self.polar2cart_mask = carth_mask 
#         else:
#             init_frame_np = init_frame.view(h, w).detach().cpu().numpy()
#             mask = (init_frame_np == 0.0).astype(np.uint8)
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#             cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#             self.polar2cart_mask = torch.tensor(cleaned_mask, device=init_frame.device, dtype=torch.float).unsqueeze(0)

#         # Base Initial Pose (Now representing the Robot's Base Link)
#         init_pose = np.array([[np.cos(init_azimuth), -np.sin(init_azimuth), init_x], 
#                               [np.sin(init_azimuth),  np.cos(init_azimuth), init_y], 
#                               [0,                     0,                    1]])
        
#         first_frame = self.polar2car(init_frame) if self.input_img_format == 'polar' else init_frame

#         self.current_pose = init_pose
#         self.sliding_window = [(first_frame, self.polar2cart_mask, init_pose)]
#         self.skip_frames = 1


#     @torch.no_grad()
#     def forward(self, frame, depth, return_visu=False):
#         new_frame = self.polar2car(frame) if self.input_img_format == 'polar' else frame
        
#         # Consensus lists
#         est_x_list = []
#         est_y_list = []
#         est_yaw_list = []
        
#         latest_visu_match = None 

#         # --- Loop over Multi-Reference Window ---
#         for i, (ref_frame, ref_mask, ref_pose) in enumerate(self.sliding_window):
#             matches = self.match_points({
#                 'image0': ref_frame, 'mask0': ref_mask,
#                 'image1': new_frame, 'mask1': self.polar2cart_mask,
#             })
            
#             pts1, pts2, confidence = matches['keypoints0'], matches['keypoints1'], matches['confidence']
#             valid_matches = confidence > self.pts_match_thresh
#             pts1, pts2 = pts1[valid_matches], pts2[valid_matches]
            
#             if len(pts1) < 3: 
#                 continue

#             pts1_r = self.scale_px2physcial(pts1)
#             pts2_r = self.scale_px2physcial(pts2)
            
#             if self.depth_compesation:
#                 ray1, ray2 = torch.sqrt(pts1_r[:, 0]**2 + pts1_r[:, 1]**2), torch.sqrt(pts2_r[:, 0]**2 + pts2_r[:, 1]**2)
#                 valid_mask = (ray1 > depth) & (ray2 > depth)
#                 pts1_r, pts2_r = pts1_r[valid_mask], pts2_r[valid_mask]
#                 ray1, ray2 = ray1[valid_mask], ray2[valid_mask]
                
#                 r1, r2 = torch.sqrt(ray1**2 - depth**2), torch.sqrt(ray2**2 - depth**2)
#                 pts1_r_scaled = pts1_r * (r1 / ray1).unsqueeze(1)
#                 pts2_r_scaled = pts2_r * (r2 / ray2).unsqueeze(1)
#             else:
#                 pts1_r_scaled, pts2_r_scaled = pts1_r, pts2_r

#             pts1_np, pts2_np = pts1_r_scaled.cpu().numpy(), pts2_r_scaled.cpu().numpy()
#             if len(pts1_np) < 3:
#                 continue

#             M, inlier_mask = cv2.estimateAffinePartial2D(
#                 pts2_np, pts1_np, method=cv2.RANSAC,
#                 ransacReprojThreshold=self.ransac_thresh, maxIters=3000, confidence=0.999,
#             )

#             if M is not None and inlier_mask is not None:
#                 inlier_mask = inlier_mask.ravel().astype(bool)
#                 inliers_abs = int(inlier_mask.sum())
#                 inliers_p = inliers_abs / pts1.shape[0] if pts1.shape[0] > 0 else 0.0

#                 if inliers_abs < 3:
#                     continue

#                 # Scale-Free Rigid Rotation Constraint
#                 angle = np.arctan2(M[1, 0], M[0, 0])
#                 R_mat_rigid = np.array([[np.cos(angle), -np.sin(angle)],
#                                         [np.sin(angle),  np.cos(angle)]])

#                 # Median Translation (No Lever Effect on Sonar Image)
#                 rotated_inliers_new = (R_mat_rigid @ pts2_np[inlier_mask].T).T
#                 diffs = pts1_np[inlier_mask] - rotated_inliers_new
#                 tx_sonar = float(np.median(diffs[:, 0]))
#                 ty_sonar = float(np.median(diffs[:, 1]))

#                 if self.ref_frame_orient == 'sim':
#                     theta = -angle
#                     tx, ty = ty_sonar, -tx_sonar
#                 elif self.ref_frame_orient == 'aracati':
#                     theta = angle
#                     tx, ty = -ty_sonar, -tx_sonar
        
#                 # 1. Delta z perspektywy samego sonaru (Lokalnie)
#                 local_translation_sonar = np.array([[ np.cos(theta), -np.sin(theta), tx],
#                                                     [ np.sin(theta),  np.cos(theta), ty], 
#                                                     [ 0,              0,             1]])

#                 # 2. KOMPENSACJA DŹWIGNI: Przeniesienie delty sonaru na środek robota
#                 local_translation_robot = self.T_R_S_2d @ local_translation_sonar @ self.T_S_R_2d

#                 # 3. Akumulacja pozycji globalnej (ref_pose = global_pose robota)
#                 est_pose = ref_pose @ local_translation_robot

#                 est_x_list.append(est_pose[0, 2])
#                 est_y_list.append(est_pose[1, 2])
#                 est_yaw_list.append(np.arctan2(est_pose[1, 0], est_pose[0, 0]))

#                 # Save metadata for visualization (preferring the latest keyframe match)
#                 if i == len(self.sliding_window) - 1 or latest_visu_match is None:
#                     latest_visu_match = {
#                         'pts1': pts1, 'pts2': pts2, 'confidence': confidence,
#                         'inliers_p': inliers_p, 'inliers_abs': inliers_abs, 'ref_frame': ref_frame,
#                         'raw_tx_sonar': tx_sonar, 'raw_ty_sonar': ty_sonar
#                     }
        
#         # --- CALCULATE CONSENSUS POSE (Robot Base Frame) ---
#         if len(est_x_list) > 0:
#             global_x = np.median(est_x_list)
#             global_y = np.median(est_y_list)
#             global_azimuth = np.arctan2(np.sum(np.sin(est_yaw_list)), np.sum(np.cos(est_yaw_list)))

#             raw_new_pose = np.array([[np.cos(global_azimuth), -np.sin(global_azimuth), global_x],
#                                      [np.sin(global_azimuth),  np.cos(global_azimuth), global_y],
#                                      [0,                       0,                      1]])
#         else:
#             raw_new_pose = self.current_pose

#         # --- DERIVE EFFECTIVE FRAME-TO-FRAME STEP (For Kinematic Gating) ---
#         # TUTAJ BYŁ BŁĄD: Liczymy krok względem ostatniej AKCEPTOWANEJ pozy, a nie KeyFrame'a.
#         R_curr = self.current_pose[0:2, 0:2]
#         t_curr = self.current_pose[0:2, 2]
#         R_new = raw_new_pose[0:2, 0:2]
#         t_new = raw_new_pose[0:2, 2]

#         R_step = R_curr.T @ R_new
#         t_step = R_curr.T @ (t_new - t_curr)

#         step_theta = float(np.arctan2(R_step[1, 0], R_step[0, 0]))
#         step_tx = float(t_step[0])
#         step_ty = float(t_step[1])

#         # --- DERIVE DISTANCE FROM KEYFRAME (For Visualisation & KeyFrame Logic) ---
#         _, _, latest_kf_pose = self.sliding_window[-1]
#         R_kf = latest_kf_pose[0:2, 0:2]
#         t_kf = latest_kf_pose[0:2, 2]

#         R_rel_kf = R_kf.T @ R_new
#         t_rel_kf = R_kf.T @ (t_new - t_kf)

#         theta_effective = float(np.arctan2(R_rel_kf[1, 0], R_rel_kf[0, 0]))
#         tx_effective = float(t_rel_kf[0])
#         ty_effective = float(t_rel_kf[1])

#         # --- SANITY CHECKS (GATING) ---
#         if not hasattr(self, 'blind_frames'):
#             self.blind_frames = 0

#         inliers_p_latest = latest_visu_match['inliers_p'] if latest_visu_match is not None else 0.0
#         inliers_abs_latest = latest_visu_match['inliers_abs'] if latest_visu_match is not None else 0

#         # 1. Statistical Gating (Twardy warunek jakości dopasowania - eliminuje halucynacje z szumu)
#         is_statistically_valid = (inliers_abs_latest >= self.min_inliers_abs) and \
#                                  (inliers_p_latest >= self.min_inliers_ratio)

#         # 2. Trust Bypass (Jeśli mamy potężną liczbę inlierów, np. >= 30, to jest to fizyczne dno, a nie szum. Ufamy bezwzględnie)
#         is_highly_trusted = (inliers_abs_latest >= 30)

#         # 3. Dynamic Kinematic Gating (Limit rośnie, jeśli opuściliśmy klatki i robot miał czas odjechać)
#         multiplier = min(self.blind_frames + 1, getattr(self, 'max_skip_frames', 3) + 1)
#         allowed_trans = self.max_trans_step * multiplier
#         allowed_rot = self.max_rot_step * multiplier

#         is_kinematically_valid = (abs(step_tx) < allowed_trans) and \
#                                  (abs(step_ty) < allowed_trans) and \
#                                  (abs(step_theta) < allowed_rot)

#         # DECYZJA (Bramkowanie)
#         step_is_valid = (len(est_x_list) > 0) and is_statistically_valid and \
#                         (is_highly_trusted or is_kinematically_valid)

#         if step_is_valid:
#             new_pose = raw_new_pose
#             self.blind_frames = 0 # Udało się, resetujemy licznik ślepoty
#         else:
#             # Fallback: Zero Velocity Model (zamrażamy trajektorię odrzucając skok w nadprzestrzeń)
#             new_pose = self.current_pose
#             self.blind_frames += 1

#         global_x, global_y = new_pose[0, 2], new_pose[1, 2]
#         global_azimuth = np.arctan2(new_pose[1, 0], new_pose[0, 0])

#         # --- KEY FRAME DETECTION ---
#         dx = global_x - latest_kf_pose[0, 2]
#         dy = global_y - latest_kf_pose[1, 2]
#         displacement = np.sqrt(dx**2 + dy**2)

#         prev_azimuth = np.arctan2(latest_kf_pose[1, 0], latest_kf_pose[0, 0])
#         azimuth_diff = np.abs(np.arctan2(np.sin(global_azimuth - prev_azimuth), np.cos(global_azimuth - prev_azimuth)))

#         key_frame_detected = False

#         # Dodajemy nową klatkę referencyjną TYLKO jeśli obecny krok jest poprawny (bezpieczna ewolucja mapy)
#         if self.key_frames and step_is_valid:
#             if (displacement >= self.key_frames_min_dist or
#                 azimuth_diff >= self.key_frames_min_rot or
#                 inliers_p_latest <= self.inliers_low_threshold):
#                 key_frame_detected = True

#         out_pose = (global_x, global_y)
#         frames_skipped_visu = self.blind_frames

#         if key_frame_detected:
#             self.sliding_window.append((new_frame, self.polar2cart_mask, new_pose))
#             if len(self.sliding_window) > self.window_size:
#                 self.sliding_window.pop(0)
#             self.current_pose = new_pose
#         else:
#             self.current_pose = new_pose

#         if not return_visu:
#             return out_pose, global_azimuth
#         else:
#             # --- Visualisation Preparation ---
#             b, c, h, w = new_frame.shape
#             if latest_visu_match is not None:
#                 frame1_np = latest_visu_match['ref_frame'].squeeze(0).permute(1, 2, 0).cpu().numpy()
#                 pts1_visu = latest_visu_match['pts1'].detach().cpu().numpy()
#                 pts2_visu = latest_visu_match['pts2'].detach().cpu().numpy()
#                 conf_visu = float(np.mean(latest_visu_match['confidence'].detach().cpu().numpy()))
#                 v_inliers_abs = latest_visu_match['inliers_abs']
#                 v_raw_tx_sonar = latest_visu_match['raw_tx_sonar']
#                 v_raw_ty_sonar = latest_visu_match['raw_ty_sonar']
#             else:
#                 frame1_np = self.sliding_window[-1][0].squeeze(0).permute(1, 2, 0).cpu().numpy()
#                 pts1_visu, pts2_visu = np.zeros((0,2)), np.zeros((0,2))
#                 conf_visu, v_inliers_abs, v_raw_tx_sonar, v_raw_ty_sonar = 0.0, 0, 0.0, 0.0

#             frame2_np = new_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             frames_np_rgb = cv2.cvtColor(np.concatenate((frame1_np, frame2_np), axis=1), cv2.COLOR_GRAY2RGB)

#             visu = {
#                 'combined_imgs': frames_np_rgb,
#                 'pts1': pts1_visu, 'pts2': pts2_visu, 'pts2_offset': (0, w),
#                 'inliers_ratio': inliers_p_latest, 'inliers_abs': v_inliers_abs,
#                 'matches_total': len(pts1_visu), 'mean_matched_confidence': conf_visu,
#                 'key_frame_detected': key_frame_detected,
#                 'step_is_valid': step_is_valid,
#                 'tx_sonar': float(v_raw_tx_sonar), 'ty_sonar': float(v_raw_ty_sonar),
#                 'tx_mapped': float(tx_effective), 'ty_mapped': float(ty_effective), 'theta': float(theta_effective),
#                 'displacement': float(displacement), 'azimuth_diff': float(azimuth_diff),
#                 'global_pose': (float(global_x), float(global_y), float(global_azimuth)),
#                 'skipped_frames': frames_skipped_visu,
#                 'window_matches_count': f"{len(est_x_list)}/{len(self.sliding_window)}",
#                 'individual_estimates': list(zip(est_x_list, est_y_list, est_yaw_list))
#             }
#             return out_pose, global_azimuth, visu
#     @torch.no_grad()
#     def polar2car(self, frame, out_shape=None):
#         out_img = F.grid_sample(frame, self.polar2cart_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
#         if self.polar2cart_mask is not None:
#             out_img = out_img * self.polar2cart_mask.unsqueeze(1)
#         return out_img

#     def scale_px2physcial(self, pts_px):
#         if pts_px.shape[0] == 0:
#             return pts_px, torch.zeros(0, dtype=torch.bool, device=self.device)
#         out_h, out_w = self.cart_frame_size
#         scale = (self.r_max - self.r_min) / out_h
#         x = (pts_px[:, 0] - out_w / 2.0) * scale
#         y = (out_h - pts_px[:, 1]) * scale + self.r_min
#         return torch.stack([x, y], dim=1)


        
#===================================================================

# import torch
# import torch.nn.functional as F
# import torch.nn as nn 
# from kornia.feature import LoFTR

# import numpy as np 
# import cv2 

# from box import Box
# import yaml

# from .utils import ExtrinsicsCalib

# class sonar_odometry(nn.Module):

#     def __init__(self, model_config, sonar_config, device, 
#                  depth_compesation=True,
#                  key_frames=True,
#                  input_img_format='polar',
#                  ref_frame_orient='sim' # 'sim', 'aracati'
#                  ):
        
#         super().__init__()

#         self.device = device 

#         self.calib = ExtrinsicsCalib(
#             T=[sonar_config.position.x, sonar_config.position.y, sonar_config.position.z],
#             R=[sonar_config.position.roll, sonar_config.position.pitch, sonar_config.position.yaw]
#         )

#         # --- 2D Lever-Arm Calibration Matrices (Sonar <-> Robot Base) ---
#         yaw_offset = sonar_config.position.yaw
#         x_offset = sonar_config.position.x
#         y_offset = sonar_config.position.y
        
#         # Matrix transforming a point in Sonar Frame to Robot Frame (T_Robot_Sonar)
#         self.T_R_S_2d = np.array([
#             [np.cos(yaw_offset), -np.sin(yaw_offset), x_offset],
#             [np.sin(yaw_offset),  np.cos(yaw_offset), y_offset],
#             [0,                   0,                  1]
#         ])
#         # Matrix transforming a point in Robot Frame to Sonar Frame (T_Sonar_Robot)
#         self.T_S_R_2d = np.linalg.inv(self.T_R_S_2d)
#         # ----------------------------------------------------------------

#         # --- init parameters --- 
#         self.ref_frame_orient = ref_frame_orient
#         self.depth_compesation = depth_compesation

#         self.key_frames = key_frames
#         self.key_frames_min_dist = model_config.key_frames_min_dist # [m]
#         self.key_frames_min_rot = model_config.key_frames_min_rot # [rad]
#         self.inliers_low_threshold = model_config.inliers_low_threshold

#         self.pts_match_thresh = model_config.pts_match_thresh # [-]
#         self.ransac_thresh = model_config.ransac_thresh # [m]

#         self.input_img_format = input_img_format
#         if self.input_img_format == 'polar':
#             self.cart_frame_size = (model_config.POLAR_FLS_INPUT_HEIGHT, 2 * model_config.POLAR_FLS_INPUT_HEIGHT)
#         else:
#             self.cart_frame_size = (model_config.CART_FLS_INPUT_HEIGHT, model_config.CART_FLS_INPUT_WIDTH)
    
#         self.r_min = sonar_config.range.min
#         self.r_max = sonar_config.range.max
#         self.theta_max = sonar_config.fov.horizontal

#         # --- init modules ---
#         pretrained = 'outdoor'
#         self.match_points = LoFTR(pretrained=pretrained).to(device).eval()

#         # --- MULTI-REFERENCE CONSENSUS WINDOW STATE --- 
#         self.window_size = 3 
#         self.sliding_window = [] # Tuples: (frame_tensor, mask_tensor, global_pose_matrix)
#         self.current_pose = None
#         self.skip_frames = 1
        
#         self.polar2cart_grid = None
#         self.polar2cart_mask = None


#     def set_init_state(self, init_x, init_y, init_azimuth, init_frame, carth_mask=None):
#         b, c, h, w = init_frame.shape
#         out_h, out_w = h, 2 * h

#         # Dynamiczne nadpisanie rozmiaru konfiguracyjnego rzeczywistymi wymiarami
#         self.cart_frame_size = (out_h, out_w)

#         # Inverse remapping grid generation
#         y = torch.arange(out_h, device=self.device, dtype=torch.float32)
#         x = torch.arange(out_w, device=self.device, dtype=torch.float32)
#         y, x = torch.meshgrid(y, x, indexing='ij')

#         x = x - out_w / 2.0
#         y = out_h - y

#         scale = (self.r_max - self.r_min) / out_h
#         x_r = x * scale
#         y_r = y * scale + self.r_min

#         r = torch.sqrt(x_r**2 + y_r**2)
#         y_r_clamp = torch.clamp(y_r, min=1e-5)
#         theta = torch.atan2(x_r, y_r_clamp)

#         norm_theta = theta / (self.theta_max / 2.0)
#         norm_r = (r - self.r_min) / (self.r_max - self.r_min) * 2.0 - 1.0

#         self.polar2cart_grid = torch.stack((norm_theta, -norm_r), dim=-1).unsqueeze(0) 

#         if self.input_img_format == 'polar':
#             valid_mask = (norm_theta >= -1.0) & (norm_theta <= 1.0) & (norm_r >= -1.0) & (norm_r <= 1.0)
#             self.polar2cart_mask = valid_mask.unsqueeze(0).expand(b, -1, -1).float()
#         elif carth_mask is not None:
#             self.polar2cart_mask = carth_mask 
#         else:
#             init_frame_np = init_frame.view(h, w).detach().cpu().numpy()
#             mask = (init_frame_np == 0.0).astype(np.uint8)
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#             cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#             self.polar2cart_mask = torch.tensor(cleaned_mask, device=init_frame.device, dtype=torch.float).unsqueeze(0)

#         # Base Initial Pose (Now representing the Robot's Base Link)
#         init_pose = np.array([[np.cos(init_azimuth), -np.sin(init_azimuth), init_x], 
#                               [np.sin(init_azimuth),  np.cos(init_azimuth), init_y], 
#                               [0,                     0,                    1]])
        
#         first_frame = self.polar2car(init_frame) if self.input_img_format == 'polar' else init_frame

#         self.current_pose = init_pose
#         self.sliding_window = [(first_frame, self.polar2cart_mask, init_pose)]
#         self.skip_frames = 1


#     @torch.no_grad()
#     def forward(self, frame, depth, return_visu=False):
#         new_frame = self.polar2car(frame) if self.input_img_format == 'polar' else frame
        
#         # Consensus lists
#         est_x_list = []
#         est_y_list = []
#         est_yaw_list = []
        
#         latest_visu_match = None 

#         # --- Loop over Multi-Reference Window ---
#         for i, (ref_frame, ref_mask, ref_pose) in enumerate(self.sliding_window):
#             matches = self.match_points({
#                 'image0': ref_frame, 'mask0': ref_mask,
#                 'image1': new_frame, 'mask1': self.polar2cart_mask,
#             })
            
#             pts1, pts2, confidence = matches['keypoints0'], matches['keypoints1'], matches['confidence']
#             valid_matches = confidence > self.pts_match_thresh
#             pts1, pts2 = pts1[valid_matches], pts2[valid_matches]
            
#             if len(pts1) < 3: 
#                 continue

#             pts1_r = self.scale_px2physcial(pts1)
#             pts2_r = self.scale_px2physcial(pts2)
            
#             if self.depth_compesation:
#                 ray1, ray2 = torch.sqrt(pts1_r[:, 0]**2 + pts1_r[:, 1]**2), torch.sqrt(pts2_r[:, 0]**2 + pts2_r[:, 1]**2)
#                 valid_mask = (ray1 > depth) & (ray2 > depth)
#                 pts1_r, pts2_r = pts1_r[valid_mask], pts2_r[valid_mask]
#                 ray1, ray2 = ray1[valid_mask], ray2[valid_mask]
                
#                 r1, r2 = torch.sqrt(ray1**2 - depth**2), torch.sqrt(ray2**2 - depth**2)
#                 pts1_r_scaled = pts1_r * (r1 / ray1).unsqueeze(1)
#                 pts2_r_scaled = pts2_r * (r2 / ray2).unsqueeze(1)
#             else:
#                 pts1_r_scaled, pts2_r_scaled = pts1_r, pts2_r

#             pts1_np, pts2_np = pts1_r_scaled.cpu().numpy(), pts2_r_scaled.cpu().numpy()
#             if len(pts1_np) < 3:
#                 continue

#             M, inlier_mask = cv2.estimateAffinePartial2D(
#                 pts2_np, pts1_np, method=cv2.RANSAC,
#                 ransacReprojThreshold=self.ransac_thresh, maxIters=3000, confidence=0.999,
#             )

#             if M is not None and inlier_mask is not None:
#                 inlier_mask = inlier_mask.ravel().astype(bool)
#                 inliers_abs = int(inlier_mask.sum())
#                 inliers_p = inliers_abs / pts1.shape[0] if pts1.shape[0] > 0 else 0.0

#                 if inliers_abs < 3:
#                     continue

#                 # Scale-Free Rigid Rotation Constraint
#                 angle = np.arctan2(M[1, 0], M[0, 0])
#                 R_mat_rigid = np.array([[np.cos(angle), -np.sin(angle)],
#                                         [np.sin(angle),  np.cos(angle)]])

#                 # Median Translation (No Lever Effect on Sonar Image)
#                 rotated_inliers_new = (R_mat_rigid @ pts2_np[inlier_mask].T).T
#                 diffs = pts1_np[inlier_mask] - rotated_inliers_new
#                 tx_sonar = float(np.median(diffs[:, 0]))
#                 ty_sonar = float(np.median(diffs[:, 1]))

#                 if self.ref_frame_orient == 'sim':
#                     theta = -angle
#                     tx, ty = ty_sonar, -tx_sonar
#                 elif self.ref_frame_orient == 'aracati':
#                     theta = angle
#                     tx, ty = -ty_sonar, -tx_sonar
        
#                 # 1. Delta z perspektywy samego sonaru (Lokalnie)
#                 local_translation_sonar = np.array([[ np.cos(theta), -np.sin(theta), tx],
#                                                     [ np.sin(theta),  np.cos(theta), ty], 
#                                                     [ 0,              0,             1]])

#                 # 2. KOMPENSACJA DŹWIGNI: Przeniesienie delty sonaru na środek robota
#                 local_translation_robot = self.T_R_S_2d @ local_translation_sonar @ self.T_S_R_2d

#                 # 3. Akumulacja pozycji globalnej (ref_pose = global_pose robota)
#                 est_pose = ref_pose @ local_translation_robot

#                 est_x_list.append(est_pose[0, 2])
#                 est_y_list.append(est_pose[1, 2])
#                 est_yaw_list.append(np.arctan2(est_pose[1, 0], est_pose[0, 0]))

#                 # Save metadata for visualization (preferring the latest keyframe match)
#                 if i == len(self.sliding_window) - 1 or latest_visu_match is None:
#                     latest_visu_match = {
#                         'pts1': pts1, 'pts2': pts2, 'confidence': confidence,
#                         'inliers_p': inliers_p, 'inliers_abs': inliers_abs, 'ref_frame': ref_frame,
#                         'raw_tx_sonar': tx_sonar, 'raw_ty_sonar': ty_sonar
#                     }

#         # --- CALCULATE CONSENSUS POSE (Robot Base Frame) ---
#         if len(est_x_list) > 0:
#             global_x = np.median(est_x_list)
#             global_y = np.median(est_y_list)
#             global_azimuth = np.arctan2(np.sum(np.sin(est_yaw_list)), np.sum(np.cos(est_yaw_list)))

#             new_pose = np.array([[np.cos(global_azimuth), -np.sin(global_azimuth), global_x],
#                                  [np.sin(global_azimuth),  np.cos(global_azimuth), global_y],
#                                  [0,                       0,                      1]])
#         else:
#             new_pose = self.current_pose
#             global_x, global_y = new_pose[0, 2], new_pose[1, 2]
#             global_azimuth = np.arctan2(new_pose[1, 0], new_pose[0, 0])

#         # --- DERIVE EFFECTIVE LOCAL TRANSFORMATION FROM CONSENSUS ---
#         _, _, latest_kf_pose = self.sliding_window[-1]
        
#         R_kf = latest_kf_pose[0:2, 0:2]
#         t_kf = latest_kf_pose[0:2, 2]
#         R_new = new_pose[0:2, 0:2]
#         t_new = new_pose[0:2, 2]

#         # Relative transformation: inv(T_kf) @ T_new (This is the effective local step of the ROBOT)
#         R_rel = R_kf.T @ R_new
#         t_rel = R_kf.T @ (t_new - t_kf)

#         theta_effective = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
#         tx_effective = float(t_rel[0])
#         ty_effective = float(t_rel[1])

#         # --- KEY FRAME DETECTION --- 
#         dx = global_x - latest_kf_pose[0, 2]
#         dy = global_y - latest_kf_pose[1, 2]
#         displacement = np.sqrt(dx**2 + dy**2)
        
#         prev_azimuth = np.arctan2(latest_kf_pose[1, 0], latest_kf_pose[0, 0])
#         azimuth_diff = np.abs(np.arctan2(np.sin(global_azimuth - prev_azimuth), np.cos(global_azimuth - prev_azimuth)))
        
#         inliers_p_latest = latest_visu_match['inliers_p'] if latest_visu_match else 0.0

#         key_frame_detected = False
#         if self.key_frames:
#             if (displacement >= self.key_frames_min_dist or 
#                 azimuth_diff >= self.key_frames_min_rot or
#                 inliers_p_latest <= self.inliers_low_threshold):
#                 key_frame_detected = True

#         out_pose = (global_x, global_y)
#         frames_skipped = self.skip_frames

#         if key_frame_detected:
#             self.sliding_window.append((new_frame, self.polar2cart_mask, new_pose))
#             if len(self.sliding_window) > self.window_size:
#                 self.sliding_window.pop(0)
#             self.current_pose = new_pose
#             self.skip_frames = 1
#         else:
#             self.current_pose = new_pose
#             self.skip_frames += 1

#         if not return_visu: 
#             return out_pose, global_azimuth
#         else: 
#             # --- Visualisation Preparation --- 
#             b, c, h, w = new_frame.shape
#             if latest_visu_match is not None:
#                 frame1_np = latest_visu_match['ref_frame'].squeeze(0).permute(1, 2, 0).cpu().numpy()
#                 pts1_visu = latest_visu_match['pts1'].detach().cpu().numpy()
#                 pts2_visu = latest_visu_match['pts2'].detach().cpu().numpy()
#                 conf_visu = float(np.mean(latest_visu_match['confidence'].detach().cpu().numpy()))
#                 v_inliers_abs = latest_visu_match['inliers_abs']
#                 v_raw_tx_sonar = latest_visu_match['raw_tx_sonar']
#                 v_raw_ty_sonar = latest_visu_match['raw_ty_sonar']
#             else:
#                 frame1_np = self.sliding_window[-1][0].squeeze(0).permute(1, 2, 0).cpu().numpy()
#                 pts1_visu, pts2_visu = np.zeros((0,2)), np.zeros((0,2))
#                 conf_visu, v_inliers_abs, v_raw_tx_sonar, v_raw_ty_sonar = 0.0, 0, 0.0, 0.0

#             frame2_np = new_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             frames_np_rgb = cv2.cvtColor(np.concatenate((frame1_np, frame2_np), axis=1), cv2.COLOR_GRAY2RGB)

#             visu = {
#                 'combined_imgs': frames_np_rgb,
#                 'pts1': pts1_visu, 'pts2': pts2_visu, 'pts2_offset': (0, w),
#                 'inliers_ratio': inliers_p_latest, 'inliers_abs': v_inliers_abs,
#                 'matches_total': len(pts1_visu), 'mean_matched_confidence': conf_visu,
#                 'key_frame_detected': key_frame_detected,
#                 'tx_sonar': float(v_raw_tx_sonar), 'ty_sonar': float(v_raw_ty_sonar),
#                 # tx_mapped is now strictly the robot's local motion to compare against GT correctly
#                 'tx_mapped': float(tx_effective), 'ty_mapped': float(ty_effective), 'theta': float(theta_effective),
#                 'displacement': float(displacement), 'azimuth_diff': float(azimuth_diff),
#                 'global_pose': (float(global_x), float(global_y), float(global_azimuth)),
#                 'skipped_frames': frames_skipped,
#                 'window_matches_count': f"{len(est_x_list)}/{len(self.sliding_window)}",
#                 'individual_estimates': list(zip(est_x_list, est_y_list, est_yaw_list))
#             }
#             return out_pose, global_azimuth, visu

#     @torch.no_grad()
#     def polar2car(self, frame, out_shape=None):
#         out_img = F.grid_sample(frame, self.polar2cart_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
#         if self.polar2cart_mask is not None:
#             out_img = out_img * self.polar2cart_mask.unsqueeze(1)
#         return out_img

#     def scale_px2physcial(self, pts_px):
#         if pts_px.shape[0] == 0:
#             return pts_px, torch.zeros(0, dtype=torch.bool, device=self.device)
#         out_h, out_w = self.cart_frame_size
#         scale = (self.r_max - self.r_min) / out_h
#         x = (pts_px[:, 0] - out_w / 2.0) * scale
#         y = (out_h - pts_px[:, 1]) * scale + self.r_min
#         return torch.stack([x, y], dim=1)

#     def fls_filter(self, frame):
#         device = frame.device 
#         _, _, c, h, w = frame.shape
#         frame_np = frame.view(h, w).unsqueeze(-1).detach().cpu().numpy()
#         blured = cv2.medianBlur(frame_np, ksize=5)
#         clahe = cv2.createCLEHE(clipLimit=2.0, tileGridSize=(8, 8))
#         filtered_frame = clahe.apply(blured)
#         return torch.tensor(filtered_frame).view(1, 1, 1, h, w).to(device)

        
# ===========================================

# import torch
# import torch.nn.functional as F
# import torch.nn as nn 
# from kornia.feature import LoFTR

# import numpy as np 
# import cv2 

# from box import Box
# import yaml

# from .utils import ExtrinsicsCalib

# class sonar_odometry(nn.Module):

#     def __init__(self, model_config, sonar_config, device, 
#                  depth_compesation=True,
#                  key_frames=True,
#                  input_img_format='polar',
#                  ref_frame_orient='sim' # 'sim', 'aracati'
#                  ):
        
#         super().__init__()

#         self.device = device 

#         self.calib = ExtrinsicsCalib(
#             T=[sonar_config.position.x, sonar_config.position.y, sonar_config.position.z],
#             R=[sonar_config.position.roll, sonar_config.position.pitch, sonar_config.position.yaw]
#         )

#         # --- init parameters --- 
#         self.ref_frame_orient = ref_frame_orient
#         self.depth_compesation = depth_compesation

#         self.key_frames = key_frames
#         self.key_frames_min_dist = model_config.key_frames_min_dist # [m]
#         self.key_frames_min_rot = model_config.key_frames_min_rot # [rad]
#         self.inliers_low_threshold = model_config.inliers_low_threshold

#         self.pts_match_thresh = model_config.pts_match_thresh # [-]
#         self.ransac_thresh = model_config.ransac_thresh # [m]

#         self.input_img_format = input_img_format
#         if self.input_img_format == 'polar':
#             self.cart_frame_size = (model_config.POLAR_FLS_INPUT_HEIGHT, 2 * model_config.POLAR_FLS_INPUT_HEIGHT)
#         else:
#             self.cart_frame_size = (model_config.CART_FLS_INPUT_HEIGHT, model_config.CART_FLS_INPUT_WIDTH)
    
#         self.r_min = sonar_config.range.min
#         self.r_max = sonar_config.range.max
#         self.theta_max = sonar_config.fov.horizontal

#         # --- init modules ---
#         pretrained = 'outdoor'
#         self.match_points = LoFTR(pretrained=pretrained).to(device).eval()

#         # --- MULTI-REFERENCE CONSENSUS WINDOW STATE --- 
#         self.window_size = 3 
#         self.sliding_window = [] # Tuples: (frame_tensor, mask_tensor, global_pose_matrix)
#         self.current_pose = None
#         self.skip_frames = 1
        
#         self.polar2cart_grid = None
#         self.polar2cart_mask = None


#     def set_init_state(self, init_x, init_y, init_azimuth, init_frame, carth_mask=None):
#         b, c, h, w = init_frame.shape
#         out_h, out_w = h, 2 * h

#         self.cart_frame_size = (out_h, out_w)

#         # Inverse remapping grid generation
#         y = torch.arange(out_h, device=self.device, dtype=torch.float32)
#         x = torch.arange(out_w, device=self.device, dtype=torch.float32)
#         y, x = torch.meshgrid(y, x, indexing='ij')

#         x = x - out_w / 2.0
#         y = out_h - y

#         scale = (self.r_max - self.r_min) / out_h
#         x_r = x * scale
#         y_r = y * scale + self.r_min

#         r = torch.sqrt(x_r**2 + y_r**2)
#         y_r_clamp = torch.clamp(y_r, min=1e-5)
#         theta = torch.atan2(x_r, y_r_clamp)

#         norm_theta = theta / (self.theta_max / 2.0)
#         norm_r = (r - self.r_min) / (self.r_max - self.r_min) * 2.0 - 1.0

#         self.polar2cart_grid = torch.stack((norm_theta, -norm_r), dim=-1).unsqueeze(0) 

#         if self.input_img_format == 'polar':
#             valid_mask = (norm_theta >= -1.0) & (norm_theta <= 1.0) & (norm_r >= -1.0) & (norm_r <= 1.0)
#             self.polar2cart_mask = valid_mask.unsqueeze(0).expand(b, -1, -1).float()
#         elif carth_mask is not None:
#             self.polar2cart_mask = carth_mask 
#         else:
#             init_frame_np = init_frame.view(h, w).detach().cpu().numpy()
#             mask = (init_frame_np == 0.0).astype(np.uint8)
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#             cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#             self.polar2cart_mask = torch.tensor(cleaned_mask, device=init_frame.device, dtype=torch.float).unsqueeze(0)

#         # Base Initial Pose
#         init_pose = np.array([[np.cos(init_azimuth), -np.sin(init_azimuth), init_x], 
#                               [np.sin(init_azimuth),  np.cos(init_azimuth), init_y], 
#                               [0,                     0,                    1]])
        
#         first_frame = self.polar2car(init_frame) if self.input_img_format == 'polar' else init_frame

#         self.current_pose = init_pose
#         self.sliding_window = [(first_frame, self.polar2cart_mask, init_pose)]
#         self.skip_frames = 1


#     @torch.no_grad()
#     def forward(self, frame, depth, return_visu=False):
#         new_frame = self.polar2car(frame) if self.input_img_format == 'polar' else frame
        
#         # Consensus lists
#         est_x_list = []
#         est_y_list = []
#         est_yaw_list = []
        
#         latest_visu_match = None 

#         # --- Loop over Multi-Reference Window ---
#         for i, (ref_frame, ref_mask, ref_pose) in enumerate(self.sliding_window):
#             matches = self.match_points({
#                 'image0': ref_frame, 'mask0': ref_mask,
#                 'image1': new_frame, 'mask1': self.polar2cart_mask,
#             })
            
#             pts1, pts2, confidence = matches['keypoints0'], matches['keypoints1'], matches['confidence']
#             valid_matches = confidence > self.pts_match_thresh
#             pts1, pts2 = pts1[valid_matches], pts2[valid_matches]
            
#             if len(pts1) < 3: 
#                 continue

#             pts1_r = self.scale_px2physcial(pts1)
#             pts2_r = self.scale_px2physcial(pts2)
            
#             if self.depth_compesation:
#                 ray1, ray2 = torch.sqrt(pts1_r[:, 0]**2 + pts1_r[:, 1]**2), torch.sqrt(pts2_r[:, 0]**2 + pts2_r[:, 1]**2)
#                 valid_mask = (ray1 > depth) & (ray2 > depth)
#                 pts1_r, pts2_r = pts1_r[valid_mask], pts2_r[valid_mask]
#                 ray1, ray2 = ray1[valid_mask], ray2[valid_mask]
                
#                 r1, r2 = torch.sqrt(ray1**2 - depth**2), torch.sqrt(ray2**2 - depth**2)
#                 pts1_r_scaled = pts1_r * (r1 / ray1).unsqueeze(1)
#                 pts2_r_scaled = pts2_r * (r2 / ray2).unsqueeze(1)
#             else:
#                 pts1_r_scaled, pts2_r_scaled = pts1_r, pts2_r

#             pts1_np, pts2_np = pts1_r_scaled.cpu().numpy(), pts2_r_scaled.cpu().numpy()
#             if len(pts1_np) < 3:
#                 continue

#             M, inlier_mask = cv2.estimateAffinePartial2D(
#                 pts2_np, pts1_np, method=cv2.RANSAC,
#                 ransacReprojThreshold=self.ransac_thresh, maxIters=3000, confidence=0.999,
#             )

#             if M is not None and inlier_mask is not None:
#                 inlier_mask = inlier_mask.ravel().astype(bool)
#                 inliers_abs = int(inlier_mask.sum())
#                 inliers_p = inliers_abs / pts1.shape[0] if pts1.shape[0] > 0 else 0.0

#                 if inliers_abs < 3:
#                     continue

#                 # Scale-Free Rigid Rotation Constraint
#                 angle = np.arctan2(M[1, 0], M[0, 0])
#                 R_mat_rigid = np.array([[np.cos(angle), -np.sin(angle)],
#                                         [np.sin(angle),  np.cos(angle)]])

#                 # Median Translation (No Lever Effect)
#                 rotated_inliers_new = (R_mat_rigid @ pts2_np[inlier_mask].T).T
#                 diffs = pts1_np[inlier_mask] - rotated_inliers_new
#                 tx_sonar = float(np.median(diffs[:, 0]))
#                 ty_sonar = float(np.median(diffs[:, 1]))

#                 if self.ref_frame_orient == 'sim':
#                     theta = -angle
#                     tx, ty = ty_sonar, -tx_sonar
#                 elif self.ref_frame_orient == 'aracati':
#                     theta = angle
#                     tx, ty = -ty_sonar, -tx_sonar
        
#                 local_translation = np.array([[ np.cos(theta), -np.sin(theta), tx],
#                                               [ np.sin(theta),  np.cos(theta), ty], 
#                                               [ 0,              0,             1]])

#                 est_pose = ref_pose @ local_translation
#                 est_x_list.append(est_pose[0, 2])
#                 est_y_list.append(est_pose[1, 2])
#                 est_yaw_list.append(np.arctan2(est_pose[1, 0], est_pose[0, 0]))

#                 # Save metadata for visualization (preferring the latest keyframe match)
#                 if i == len(self.sliding_window) - 1 or latest_visu_match is None:
#                     latest_visu_match = {
#                         'pts1': pts1, 'pts2': pts2, 'confidence': confidence,
#                         'inliers_p': inliers_p, 'inliers_abs': inliers_abs, 'ref_frame': ref_frame
#                     }

#         # --- CALCULATE CONSENSUS POSE ---
#         if len(est_x_list) > 0:
#             global_x = np.median(est_x_list)
#             global_y = np.median(est_y_list)
#             global_azimuth = np.arctan2(np.sum(np.sin(est_yaw_list)), np.sum(np.cos(est_yaw_list)))

#             new_pose = np.array([[np.cos(global_azimuth), -np.sin(global_azimuth), global_x],
#                                  [np.sin(global_azimuth),  np.cos(global_azimuth), global_y],
#                                  [0,                       0,                      1]])
#         else:
#             new_pose = self.current_pose
#             global_x, global_y = new_pose[0, 2], new_pose[1, 2]
#             global_azimuth = np.arctan2(new_pose[1, 0], new_pose[0, 0])

#         # --- DERIVE EFFECTIVE LOCAL TRANSFORMATION FROM CONSENSUS ---
#         # Back-calculating local step relative to the LATEST keyframe for true logging consistency
#         _, _, latest_kf_pose = self.sliding_window[-1]
        
#         R_kf = latest_kf_pose[0:2, 0:2]
#         t_kf = latest_kf_pose[0:2, 2]
#         R_new = new_pose[0:2, 0:2]
#         t_new = new_pose[0:2, 2]

#         # Relative transformation: inv(T_kf) @ T_new
#         R_rel = R_kf.T @ R_new
#         t_rel = R_kf.T @ (t_new - t_kf)

#         theta_effective = np.arctan2(R_rel[1, 0], R_rel[0, 0])
#         tx_effective = t_rel[0]
#         ty_effective = t_rel[1]

#         # Re-map back to sonar physical coordinates for visualization consistency
#         if self.ref_frame_orient == 'sim':
#             tx_sonar_eff = -ty_effective
#             ty_sonar_eff = tx_effective
#         elif self.ref_frame_orient == 'aracati':
#             tx_sonar_eff = -ty_effective
#             ty_sonar_eff = -tx_effective
#         else:
#             tx_sonar_eff, ty_sonar_eff = tx_effective, ty_effective

#         # --- KEY FRAME DETECTION --- 
#         dx = global_x - latest_kf_pose[0, 2]
#         dy = global_y - latest_kf_pose[1, 2]
#         displacement = np.sqrt(dx**2 + dy**2)
        
#         prev_azimuth = np.arctan2(latest_kf_pose[1, 0], latest_kf_pose[0, 0])
#         azimuth_diff = np.abs(np.arctan2(np.sin(global_azimuth - prev_azimuth), np.cos(global_azimuth - prev_azimuth)))
        
#         inliers_p_latest = latest_visu_match['inliers_p'] if latest_visu_match else 0.0

#         key_frame_detected = False
#         if self.key_frames:
#             if (displacement >= self.key_frames_min_dist or 
#                 azimuth_diff >= self.key_frames_min_rot or
#                 inliers_p_latest <= self.inliers_low_threshold):
#                 key_frame_detected = True

#         out_pose = (global_x, global_y)
#         frames_skipped = self.skip_frames

#         if key_frame_detected:
#             self.sliding_window.append((new_frame, self.polar2cart_mask, new_pose))
#             if len(self.sliding_window) > self.window_size:
#                 self.sliding_window.pop(0)
#             self.current_pose = new_pose
#             self.skip_frames = 1
#         else:
#             self.current_pose = new_pose
#             self.skip_frames += 1

#         if not return_visu: 
#             return out_pose, global_azimuth
#         else: 
#             # --- Visualisation Preparation --- 
#             b, c, h, w = new_frame.shape
#             if latest_visu_match is not None:
#                 frame1_np = latest_visu_match['ref_frame'].squeeze(0).permute(1, 2, 0).cpu().numpy()
#                 pts1_visu = latest_visu_match['pts1'].detach().cpu().numpy()
#                 pts2_visu = latest_visu_match['pts2'].detach().cpu().numpy()
#                 conf_visu = float(np.mean(latest_visu_match['confidence'].detach().cpu().numpy()))
#                 v_inliers_abs = latest_visu_match['inliers_abs']
#             else:
#                 frame1_np = self.sliding_window[-1][0].squeeze(0).permute(1, 2, 0).cpu().numpy()
#                 pts1_visu, pts2_visu = np.zeros((0,2)), np.zeros((0,2))
#                 conf_visu, v_inliers_abs = 0.0, 0

#             frame2_np = new_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             frames_np_rgb = cv2.cvtColor(np.concatenate((frame1_np, frame2_np), axis=1), cv2.COLOR_GRAY2RGB)

#             visu = {
#                 'combined_imgs': frames_np_rgb,
#                 'pts1': pts1_visu, 'pts2': pts2_visu, 'pts2_offset': (0, w),
#                 'inliers_ratio': inliers_p_latest, 'inliers_abs': v_inliers_abs,
#                 'matches_total': len(pts1_visu), 'mean_matched_confidence': conf_visu,
#                 'key_frame_detected': key_frame_detected,
#                 'tx_sonar': float(tx_sonar_eff), 'ty_sonar': float(ty_sonar_eff),
#                 'tx_mapped': float(tx_effective), 'ty_mapped': float(ty_effective), 'theta': float(theta_effective),
#                 'displacement': float(displacement), 'azimuth_diff': float(azimuth_diff),
#                 'global_pose': (float(global_x), float(global_y), float(global_azimuth)),
#                 'skipped_frames': frames_skipped,
#                 # --- NEW DEBUG METRICS ---
#                 'window_matches_count': f"{len(est_x_list)}/{len(self.sliding_window)}",
#                 'individual_estimates': list(zip(est_x_list, est_y_list, est_yaw_list))
#             }
#             return out_pose, global_azimuth, visu

#     @torch.no_grad()
#     def polar2car(self, frame, out_shape=None):
#         out_img = F.grid_sample(frame, self.polar2cart_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
#         if self.polar2cart_mask is not None:
#             out_img = out_img * self.polar2cart_mask.unsqueeze(1)
#         return out_img

#     def scale_px2physcial(self, pts_px):
#         if pts_px.shape[0] == 0:
#             return pts_px, torch.zeros(0, dtype=torch.bool, device=self.device)
#         out_h, out_w = self.cart_frame_size
#         scale = (self.r_max - self.r_min) / out_h
#         x = (pts_px[:, 0] - out_w / 2.0) * scale
#         y = (out_h - pts_px[:, 1]) * scale + self.r_min
#         return torch.stack([x, y], dim=1)

#     def fls_filter(self, frame):
#         device = frame.device 
#         _, _, c, h, w = frame.shape
#         frame_np = frame.view(h, w).unsqueeze(-1).detach().cpu().numpy()
#         blured = cv2.medianBlur(frame_np, ksize=5)
#         clahe = cv2.createCLEHE(clipLimit=2.0, tileGridSize=(8, 8))
#         filtered_frame = clahe.apply(blured)
#         return torch.tensor(filtered_frame).view(1, 1, 1, h, w).to(device)


