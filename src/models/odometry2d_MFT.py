import torch
import torch.nn.functional as F
import torch.nn as nn 
from kornia.feature import LoFTR

import numpy as np 
import cv2 

from box import Box
import yaml

from .utils import ExtrinsicsCalib

class sonar_odometry(nn.Module):

    def __init__(self, model_config, sonar_config, device, 
                 depth_compesation=True,
                 key_frames=True,
                 input_img_format='polar',
                 ref_frame_orient='sim' # 'sim', 'aracati'
                 ):
        
        super().__init__()

        self.device = device 

        self.calib = ExtrinsicsCalib(
            T=[sonar_config.position.x, sonar_config.position.y, sonar_config.position.z],
            R=[sonar_config.position.roll, sonar_config.position.pitch, sonar_config.position.yaw]
        )

        # --- init parameters --- 
        self.ref_frame_orient = ref_frame_orient
        self.depth_compesation = depth_compesation

        self.key_frames = key_frames
        self.key_frames_min_dist = model_config.key_frames_min_dist # [m]
        self.key_frames_min_rot = model_config.key_frames_min_rot # [rad]
        self.inliers_low_threshold = model_config.inliers_low_threshold

        self.pts_match_thresh = model_config.pts_match_thresh # [-]
        self.ransac_thresh = model_config.ransac_thresh # [m]

        self.input_img_format = input_img_format
        if self.input_img_format == 'polar':
            self.cart_frame_size = (model_config.POLAR_FLS_INPUT_HEIGHT, 2 * model_config.POLAR_FLS_INPUT_HEIGHT)
        else:
            self.cart_frame_size = (model_config.CART_FLS_INPUT_HEIGHT, model_config.CART_FLS_INPUT_WIDTH)
    
        self.r_min = sonar_config.range.min
        self.r_max = sonar_config.range.max
        self.theta_max = sonar_config.fov.horizontal

        # --- init modules ---
        pretrained = 'outdoor'
        self.match_points = LoFTR(pretrained=pretrained).to(device).eval()

        # --- MULTI-REFERENCE CONSENSUS WINDOW STATE --- 
        self.window_size = 3 
        self.sliding_window = [] # Tuples: (frame_tensor, mask_tensor, global_pose_matrix)
        self.current_pose = None
        self.skip_frames = 1
        
        self.polar2cart_grid = None
        self.polar2cart_mask = None


    def set_init_state(self, init_x, init_y, init_azimuth, init_frame, carth_mask=None):
        b, c, h, w = init_frame.shape
        out_h, out_w = h, 2 * h

        self.cart_frame_size = (out_h, out_w)

        # Inverse remapping grid generation
        y = torch.arange(out_h, device=self.device, dtype=torch.float32)
        x = torch.arange(out_w, device=self.device, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing='ij')

        x = x - out_w / 2.0
        y = out_h - y

        scale = (self.r_max - self.r_min) / out_h
        x_r = x * scale
        y_r = y * scale + self.r_min

        r = torch.sqrt(x_r**2 + y_r**2)
        y_r_clamp = torch.clamp(y_r, min=1e-5)
        theta = torch.atan2(x_r, y_r_clamp)

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

        # Base Initial Pose
        init_pose = np.array([[np.cos(init_azimuth), -np.sin(init_azimuth), init_x], 
                              [np.sin(init_azimuth),  np.cos(init_azimuth), init_y], 
                              [0,                     0,                    1]])
        
        first_frame = self.polar2car(init_frame) if self.input_img_format == 'polar' else init_frame

        self.current_pose = init_pose
        self.sliding_window = [(first_frame, self.polar2cart_mask, init_pose)]
        self.skip_frames = 1


    @torch.no_grad()
    def forward(self, frame, depth, return_visu=False):
        new_frame = self.polar2car(frame) if self.input_img_format == 'polar' else frame
        
        # Consensus lists
        est_x_list = []
        est_y_list = []
        est_yaw_list = []
        
        latest_visu_match = None 

        # --- Loop over Multi-Reference Window ---
        for i, (ref_frame, ref_mask, ref_pose) in enumerate(self.sliding_window):
            matches = self.match_points({
                'image0': ref_frame, 'mask0': ref_mask,
                'image1': new_frame, 'mask1': self.polar2cart_mask,
            })
            
            pts1, pts2, confidence = matches['keypoints0'], matches['keypoints1'], matches['confidence']
            valid_matches = confidence > self.pts_match_thresh
            pts1, pts2 = pts1[valid_matches], pts2[valid_matches]
            
            if len(pts1) < 3: 
                continue

            pts1_r = self.scale_px2physcial(pts1)
            pts2_r = self.scale_px2physcial(pts2)
            
            if self.depth_compesation:
                ray1, ray2 = torch.sqrt(pts1_r[:, 0]**2 + pts1_r[:, 1]**2), torch.sqrt(pts2_r[:, 0]**2 + pts2_r[:, 1]**2)
                valid_mask = (ray1 > depth) & (ray2 > depth)
                pts1_r, pts2_r = pts1_r[valid_mask], pts2_r[valid_mask]
                ray1, ray2 = ray1[valid_mask], ray2[valid_mask]
                
                r1, r2 = torch.sqrt(ray1**2 - depth**2), torch.sqrt(ray2**2 - depth**2)
                pts1_r_scaled = pts1_r * (r1 / ray1).unsqueeze(1)
                pts2_r_scaled = pts2_r * (r2 / ray2).unsqueeze(1)
            else:
                pts1_r_scaled, pts2_r_scaled = pts1_r, pts2_r

            pts1_np, pts2_np = pts1_r_scaled.cpu().numpy(), pts2_r_scaled.cpu().numpy()
            if len(pts1_np) < 3:
                continue

            M, inlier_mask = cv2.estimateAffinePartial2D(
                pts2_np, pts1_np, method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh, maxIters=3000, confidence=0.999,
            )

            if M is not None and inlier_mask is not None:
                inlier_mask = inlier_mask.ravel().astype(bool)
                inliers_abs = int(inlier_mask.sum())
                inliers_p = inliers_abs / pts1.shape[0] if pts1.shape[0] > 0 else 0.0

                if inliers_abs < 3:
                    continue

                # Scale-Free Rigid Rotation Constraint
                angle = np.arctan2(M[1, 0], M[0, 0])
                R_mat_rigid = np.array([[np.cos(angle), -np.sin(angle)],
                                        [np.sin(angle),  np.cos(angle)]])

                # Median Translation (No Lever Effect)
                rotated_inliers_new = (R_mat_rigid @ pts2_np[inlier_mask].T).T
                diffs = pts1_np[inlier_mask] - rotated_inliers_new
                tx_sonar = float(np.median(diffs[:, 0]))
                ty_sonar = float(np.median(diffs[:, 1]))

                if self.ref_frame_orient == 'sim':
                    theta = -angle
                    tx, ty = ty_sonar, -tx_sonar
                elif self.ref_frame_orient == 'aracati':
                    theta = angle
                    tx, ty = -ty_sonar, -tx_sonar
        
                local_translation = np.array([[ np.cos(theta), -np.sin(theta), tx],
                                              [ np.sin(theta),  np.cos(theta), ty], 
                                              [ 0,              0,             1]])

                est_pose = ref_pose @ local_translation
                est_x_list.append(est_pose[0, 2])
                est_y_list.append(est_pose[1, 2])
                est_yaw_list.append(np.arctan2(est_pose[1, 0], est_pose[0, 0]))

                # Save metadata for visualization (preferring the latest keyframe match)
                if i == len(self.sliding_window) - 1 or latest_visu_match is None:
                    latest_visu_match = {
                        'pts1': pts1, 'pts2': pts2, 'confidence': confidence,
                        'inliers_p': inliers_p, 'inliers_abs': inliers_abs, 'ref_frame': ref_frame
                    }

        # --- CALCULATE CONSENSUS POSE ---
        if len(est_x_list) > 0:
            global_x = np.median(est_x_list)
            global_y = np.median(est_y_list)
            global_azimuth = np.arctan2(np.sum(np.sin(est_yaw_list)), np.sum(np.cos(est_yaw_list)))

            new_pose = np.array([[np.cos(global_azimuth), -np.sin(global_azimuth), global_x],
                                 [np.sin(global_azimuth),  np.cos(global_azimuth), global_y],
                                 [0,                       0,                      1]])
        else:
            new_pose = self.current_pose
            global_x, global_y = new_pose[0, 2], new_pose[1, 2]
            global_azimuth = np.arctan2(new_pose[1, 0], new_pose[0, 0])

        # --- DERIVE EFFECTIVE LOCAL TRANSFORMATION FROM CONSENSUS ---
        # Back-calculating local step relative to the LATEST keyframe for true logging consistency
        _, _, latest_kf_pose = self.sliding_window[-1]
        
        R_kf = latest_kf_pose[0:2, 0:2]
        t_kf = latest_kf_pose[0:2, 2]
        R_new = new_pose[0:2, 0:2]
        t_new = new_pose[0:2, 2]

        # Relative transformation: inv(T_kf) @ T_new
        R_rel = R_kf.T @ R_new
        t_rel = R_kf.T @ (t_new - t_kf)

        theta_effective = np.arctan2(R_rel[1, 0], R_rel[0, 0])
        tx_effective = t_rel[0]
        ty_effective = t_rel[1]

        # Re-map back to sonar physical coordinates for visualization consistency
        if self.ref_frame_orient == 'sim':
            tx_sonar_eff = -ty_effective
            ty_sonar_eff = tx_effective
        elif self.ref_frame_orient == 'aracati':
            tx_sonar_eff = -ty_effective
            ty_sonar_eff = -tx_effective
        else:
            tx_sonar_eff, ty_sonar_eff = tx_effective, ty_effective

        # --- KEY FRAME DETECTION --- 
        dx = global_x - latest_kf_pose[0, 2]
        dy = global_y - latest_kf_pose[1, 2]
        displacement = np.sqrt(dx**2 + dy**2)
        
        prev_azimuth = np.arctan2(latest_kf_pose[1, 0], latest_kf_pose[0, 0])
        azimuth_diff = np.abs(np.arctan2(np.sin(global_azimuth - prev_azimuth), np.cos(global_azimuth - prev_azimuth)))
        
        inliers_p_latest = latest_visu_match['inliers_p'] if latest_visu_match else 0.0

        key_frame_detected = False
        if self.key_frames:
            if (displacement >= self.key_frames_min_dist or 
                azimuth_diff >= self.key_frames_min_rot or
                inliers_p_latest <= self.inliers_low_threshold):
                key_frame_detected = True

        out_pose = (global_x, global_y)
        frames_skipped = self.skip_frames

        if key_frame_detected:
            self.sliding_window.append((new_frame, self.polar2cart_mask, new_pose))
            if len(self.sliding_window) > self.window_size:
                self.sliding_window.pop(0)
            self.current_pose = new_pose
            self.skip_frames = 1
        else:
            self.current_pose = new_pose
            self.skip_frames += 1

        if not return_visu: 
            return out_pose, global_azimuth
        else: 
            # --- Visualisation Preparation --- 
            b, c, h, w = new_frame.shape
            if latest_visu_match is not None:
                frame1_np = latest_visu_match['ref_frame'].squeeze(0).permute(1, 2, 0).cpu().numpy()
                pts1_visu = latest_visu_match['pts1'].detach().cpu().numpy()
                pts2_visu = latest_visu_match['pts2'].detach().cpu().numpy()
                conf_visu = float(np.mean(latest_visu_match['confidence'].detach().cpu().numpy()))
                v_inliers_abs = latest_visu_match['inliers_abs']
            else:
                frame1_np = self.sliding_window[-1][0].squeeze(0).permute(1, 2, 0).cpu().numpy()
                pts1_visu, pts2_visu = np.zeros((0,2)), np.zeros((0,2))
                conf_visu, v_inliers_abs = 0.0, 0

            frame2_np = new_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frames_np_rgb = cv2.cvtColor(np.concatenate((frame1_np, frame2_np), axis=1), cv2.COLOR_GRAY2RGB)

            visu = {
                'combined_imgs': frames_np_rgb,
                'pts1': pts1_visu, 'pts2': pts2_visu, 'pts2_offset': (0, w),
                'inliers_ratio': inliers_p_latest, 'inliers_abs': v_inliers_abs,
                'matches_total': len(pts1_visu), 'mean_matched_confidence': conf_visu,
                'key_frame_detected': key_frame_detected,
                'tx_sonar': float(tx_sonar_eff), 'ty_sonar': float(ty_sonar_eff),
                'tx_mapped': float(tx_effective), 'ty_mapped': float(ty_effective), 'theta': float(theta_effective),
                'displacement': float(displacement), 'azimuth_diff': float(azimuth_diff),
                'global_pose': (float(global_x), float(global_y), float(global_azimuth)),
                'skipped_frames': frames_skipped,
                # --- NEW DEBUG METRICS ---
                'window_matches_count': f"{len(est_x_list)}/{len(self.sliding_window)}",
                'individual_estimates': list(zip(est_x_list, est_y_list, est_yaw_list))
            }
            return out_pose, global_azimuth, visu

    @torch.no_grad()
    def polar2car(self, frame, out_shape=None):
        out_img = F.grid_sample(frame, self.polar2cart_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        if self.polar2cart_mask is not None:
            out_img = out_img * self.polar2cart_mask.unsqueeze(1)
        return out_img

    def scale_px2physcial(self, pts_px):
        if pts_px.shape[0] == 0:
            return pts_px, torch.zeros(0, dtype=torch.bool, device=self.device)
        out_h, out_w = self.cart_frame_size
        scale = (self.r_max - self.r_min) / out_h
        x = (pts_px[:, 0] - out_w / 2.0) * scale
        y = (out_h - pts_px[:, 1]) * scale + self.r_min
        return torch.stack([x, y], dim=1)

    def fls_filter(self, frame):
        device = frame.device 
        _, _, c, h, w = frame.shape
        frame_np = frame.view(h, w).unsqueeze(-1).detach().cpu().numpy()
        blured = cv2.medianBlur(frame_np, ksize=5)
        clahe = cv2.createCLEHE(clipLimit=2.0, tileGridSize=(8, 8))
        filtered_frame = clahe.apply(blured)
        return torch.tensor(filtered_frame).view(1, 1, 1, h, w).to(device)
    
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
#         self.window_size = 3 # N frames in sliding window
#         self.sliding_window = [] # Stores tuples: (frame_tensor, mask_tensor, global_pose_matrix)
#         self.current_pose = None
#         self.skip_frames = 1
        
#         self.polar2cart_grid = None
#         self.polar2cart_mask = None


#     def set_init_state(self, init_x, init_y, init_azimuth, init_frame, carth_mask=None):

#         # --- generate sampling grid once to speed up ---
#         b, c, h, w = init_frame.shape
        
#         # set output shape
#         out_h = h
#         out_w = 2 * h

#         # Inverse remapping 
#         y = torch.arange(out_h, device=self.device, dtype=torch.float32)
#         x = torch.arange(out_w, device=self.device, dtype=torch.float32)
#         y, x = torch.meshgrid(y, x, indexing='ij')

#         # Recenter
#         x = x - out_w / 2.0
#         y = out_h - y

#         # Rescale to real-world values (metry)
#         scale = (self.r_max - self.r_min) / out_h
#         x_r = x * scale
#         y_r = y * scale + self.r_min

#         # Map (x, y) -> (theta, r)
#         r = torch.sqrt(x_r**2 + y_r**2)
#         y_r_clamp = torch.clamp(y_r, min=1e-5)
#         theta = torch.atan2(x_r, y_r_clamp)

#         # Normalization 
#         norm_theta = theta / (self.theta_max / 2.0)
#         norm_r = (r - self.r_min) / (self.r_max - self.r_min) * 2.0 - 1.0

#         # Create grid with shape (b, out_h, out_w, 2)
#         self.polar2cart_grid = torch.stack((norm_theta, -norm_r), dim=-1).unsqueeze(0) 

#         # Create valid pixels mask 
#         if self.input_img_format == 'polar':
#             valid_mask = (norm_theta >= -1.0) & (norm_theta <= 1.0) & (norm_r >= -1.0) & (norm_r <= 1.0)
#             self.polar2cart_mask = valid_mask.unsqueeze(0).expand(b, -1, -1).float()
#         elif carth_mask is not None:
#             self.polar2cart_mask = carth_mask 
#         else:
#             init_frame_np = init_frame.view(h, w).detach().cpu().numpy()
#             mask = (init_frame_np == 0.0).astype(np.uint8)
#             # morph clean
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#             cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#             mask_torch = torch.tensor(cleaned_mask, device=init_frame.device, dtype=torch.float)
#             self.polar2cart_mask = mask_torch.unsqueeze(0)

#         # --- save first data as init state ---
#         init_pose = np.array([[np.cos(init_azimuth), -np.sin(init_azimuth), init_x], 
#                               [np.sin(init_azimuth),  np.cos(init_azimuth), init_y], 
#                               [0,                     0,                    1]])
        
#         if self.input_img_format == 'polar':
#             first_frame = self.polar2car(init_frame)
#         else:
#             first_frame = init_frame

#         # Initialize tracking states
#         self.current_pose = init_pose
#         self.sliding_window = [(first_frame, self.polar2cart_mask, init_pose)]
#         self.skip_frames = 1


#     @torch.no_grad()
#     def forward(self, frame, depth, return_visu=False):

#         # --- convert new frame to carthesian ---
#         if self.input_img_format == 'polar':
#             new_frame = self.polar2car(frame)
#         else: 
#             new_frame = frame
        
#         # Consensus lists
#         est_x_list = []
#         est_y_list = []
#         est_yaw_list = []
        
#         # Var to store best/latest visualization data
#         latest_visu_data = None 

#         # --- Loop over Sliding Window ---
#         for i, (ref_frame, ref_mask, ref_pose) in enumerate(self.sliding_window):
            
#             # --- match points with loftr ---
#             matches = self.match_points({
#                 'image0': ref_frame, 'mask0': ref_mask,
#                 'image1': new_frame, 'mask1': self.polar2cart_mask,
#             })
            
#             pts1 = matches['keypoints0']
#             pts2 = matches['keypoints1']
#             confidence = matches['confidence']

#             # filter with matching confidence 
#             valid_matches = confidence > self.pts_match_thresh
#             pts1 = pts1[valid_matches]
#             pts2 = pts2[valid_matches]
            
#             if len(pts1) < 3: # Not enough points for RANSAC
#                 continue

#             # transform to real-world values
#             pts1_r = self.scale_px2physcial(pts1)
#             pts2_r = self.scale_px2physcial(pts2)
            
#             # --- compensate depth change ---
#             if self.depth_compesation:
#                 ray1 = torch.sqrt(pts1_r[:, 0]**2 + pts1_r[:, 1]**2)
#                 ray2 = torch.sqrt(pts2_r[:, 0]**2 + pts2_r[:, 1]**2)

#                 valid_mask = (ray1 > depth) & (ray2 > depth)
#                 pts1_r = pts1_r[valid_mask]
#                 pts2_r = pts2_r[valid_mask]
#                 ray1 = ray1[valid_mask]
#                 ray2 = ray2[valid_mask]
                
#                 r1 = torch.sqrt(ray1**2 - depth**2)
#                 r2 = torch.sqrt(ray2**2 - depth**2)

#                 pts1_r_scaled = pts1_r * (r1 / ray1).unsqueeze(1)
#                 pts2_r_scaled = pts2_r * (r2 / ray2).unsqueeze(1)
#             else:
#                 pts1_r_scaled = pts1_r
#                 pts2_r_scaled = pts2_r

#             pts1_np = pts1_r_scaled.cpu().numpy()
#             pts2_np = pts2_r_scaled.cpu().numpy()

#             if len(pts1_np) < 3:
#                 continue

#             # --- extract transform matrix - RANSAC ---  
#             M, inlier_mask = cv2.estimateAffinePartial2D(
#                 pts2_np, pts1_np,
#                 method=cv2.RANSAC,
#                 ransacReprojThreshold=self.ransac_thresh,   
#                 maxIters=3000,
#                 confidence=0.999,
#             )

#             if M is not None and inlier_mask is not None:
#                 inlier_mask = inlier_mask.ravel().astype(bool)
#                 inliers_abs = int(inlier_mask.sum())
#                 inliers_p = inliers_abs / pts1.shape[0] if pts1.shape[0] > 0 else 0.0

#                 if inliers_abs < 3:
#                     continue

#                 # Scale-Free Rotation (Rigid constraint)
#                 # cv2 M is [2, 3] mapping pts2 to pts1.
#                 angle = np.arctan2(M[1, 0], M[0, 0])
#                 R_mat_rigid = np.array([
#                     [np.cos(angle), -np.sin(angle)],
#                     [np.sin(angle),  np.cos(angle)]
#                 ])

#                 # Extract inliers
#                 inliers_ref = pts1_np[inlier_mask]
#                 inliers_new = pts2_np[inlier_mask]

#                 # Median Translation (Prevents Lever Effect)
#                 rotated_inliers_new = (R_mat_rigid @ inliers_new.T).T
#                 diffs = inliers_ref - rotated_inliers_new
                
#                 tx_sonar = float(np.median(diffs[:, 0]))
#                 ty_sonar = float(np.median(diffs[:, 1]))

#                 # Map Axes Based on Orientation
#                 if self.ref_frame_orient == 'sim':
#                     theta = -angle
#                     tx = ty_sonar
#                     ty = -tx_sonar
#                 elif self.ref_frame_orient == 'aracati':
#                     theta = angle
#                     tx = -ty_sonar
#                     ty = -tx_sonar
        
#                 local_translation = np.array([[ np.cos(theta), -np.sin(theta), tx],
#                                               [ np.sin(theta),  np.cos(theta), ty], 
#                                               [ 0,              0,             1]])

#                 est_pose = ref_pose @ local_translation
                
#                 est_x_list.append(est_pose[0, 2])
#                 est_y_list.append(est_pose[1, 2])
#                 est_yaw_list.append(np.arctan2(est_pose[1, 0], est_pose[0, 0]))

#                 # Save visu for the most recent keyframe in the window (usually the last element)
#                 if i == len(self.sliding_window) - 1:
#                     latest_visu_data = {
#                         'pts1': pts1, 'pts2': pts2, 'confidence': confidence,
#                         'inliers_p': inliers_p, 'inliers_abs': inliers_abs,
#                         'tx_sonar': tx_sonar, 'ty_sonar': ty_sonar,
#                         'tx': tx, 'ty': ty, 'theta': theta,
#                         'ref_frame': ref_frame
#                     }

#         # --- CALCULATE CONSENSUS ---
#         if len(est_x_list) > 0:
#             # Median for translation
#             global_x = np.median(est_x_list)
#             global_y = np.median(est_y_list)
            
#             # Directional Mean for Yaw (prevents issues around -pi/pi)
#             sum_sin = np.sum(np.sin(est_yaw_list))
#             sum_cos = np.sum(np.cos(est_yaw_list))
#             global_azimuth = np.arctan2(sum_sin, sum_cos)

#             new_pose = np.array([[np.cos(global_azimuth), -np.sin(global_azimuth), global_x],
#                                  [np.sin(global_azimuth),  np.cos(global_azimuth), global_y],
#                                  [0,                       0,                      1]])
#         else:
#             # Fallback if ALL matches in the window fail
#             new_pose = self.current_pose
#             global_x = new_pose[0, 2]
#             global_y = new_pose[1, 2]
#             global_azimuth = np.arctan2(new_pose[1, 0], new_pose[0, 0])


#         # --- KEY FRAME DETECTION --- 
#         # Evaluate displacement against the LATEST keyframe in the window
#         latest_ref_pose = self.sliding_window[-1][2]
#         dx = global_x - latest_ref_pose[0, 2]
#         dy = global_y - latest_ref_pose[1, 2]
#         displacement = np.sqrt(dx**2 + dy**2)
        
#         prev_azimuth = np.arctan2(latest_ref_pose[1, 0], latest_ref_pose[0, 0])
#         azimuth_diff = np.abs(np.arctan2(np.sin(global_azimuth - prev_azimuth), 
#                                          np.cos(global_azimuth - prev_azimuth)))
        
#         inliers_p_latest = latest_visu_data['inliers_p'] if latest_visu_data else 0.0

#         key_frame_detected = False
#         if self.key_frames:
#             if (displacement >= self.key_frames_min_dist or 
#                 azimuth_diff >= self.key_frames_min_rot or
#                 inliers_p_latest <= self.inliers_low_threshold):
#                 key_frame_detected = True

#         out_pose = (global_x, global_y)
        
#         # --- Update Sliding Window ---
#         frames_skipped = self.skip_frames

#         if key_frame_detected:
#             self.sliding_window.append((new_frame, self.polar2cart_mask, new_pose))
#             # FIFO: Remove oldest if we exceed window size
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
#             # --- visualisation --- 
#             b, c, h, w = new_frame.shape
            
#             # Use latest ref frame for visu, or blank if nothing matched
#             if latest_visu_data is not None:
#                 frame1_np = latest_visu_data['ref_frame'].squeeze(0).permute(1, 2, 0).cpu().numpy()
#                 pts1_visu = latest_visu_data['pts1'].detach().cpu().numpy()
#                 pts2_visu = latest_visu_data['pts2'].detach().cpu().numpy()
#                 conf_visu = float(np.mean(latest_visu_data['confidence'].detach().cpu().numpy()))
#                 v_tx_sonar, v_ty_sonar = latest_visu_data['tx_sonar'], latest_visu_data['ty_sonar']
#                 v_tx, v_ty, v_theta = latest_visu_data['tx'], latest_visu_data['ty'], latest_visu_data['theta']
#                 v_inliers_abs = latest_visu_data['inliers_abs']
#             else:
#                 frame1_np = self.sliding_window[-1][0].squeeze(0).permute(1, 2, 0).cpu().numpy()
#                 pts1_visu, pts2_visu = np.zeros((0,2)), np.zeros((0,2))
#                 conf_visu, v_tx_sonar, v_ty_sonar, v_tx, v_ty, v_theta, v_inliers_abs = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

#             frame2_np = new_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             frames_np = np.concatenate((frame1_np, frame2_np), axis=1)
#             frames_np_rgb = cv2.cvtColor(frames_np, cv2.COLOR_GRAY2RGB)

#             visu = {
#                 'combined_imgs': frames_np_rgb,
#                 'pts1': pts1_visu,
#                 'pts2': pts2_visu,
#                 'pts2_offset': (0, w),
#                 'inliers_ratio': inliers_p_latest,
#                 'inliers_abs': v_inliers_abs,
#                 'matches_total': len(pts1_visu),
#                 'mean_matched_confidence': conf_visu,
#                 'key_frame_detected': key_frame_detected,
#                 'tx_sonar': float(v_tx_sonar),
#                 'ty_sonar': float(v_ty_sonar),
#                 'tx_mapped': float(v_tx),
#                 'ty_mapped': float(v_ty),
#                 'theta': float(v_theta),
#                 'displacement': float(displacement),
#                 'azimuth_diff': float(azimuth_diff),
#                 'global_pose': (float(global_x), float(global_y), float(global_azimuth)),
#                 'skipped_frames': frames_skipped
#             }

#             return out_pose, global_azimuth, visu

#     @torch.no_grad()
#     def polar2car(self, frame, out_shape=None):
#         # Sample pixels with grid, padd with zeros
#         out_img = F.grid_sample(frame, self.polar2cart_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

#         if self.polar2cart_mask is not None:
#             out_img = out_img * self.polar2cart_mask.unsqueeze(1)
            
#         return out_img

#     def scale_px2physcial(self, pts_px):
#         if pts_px.shape[0] == 0:
#             return pts_px, torch.zeros(0, dtype=torch.bool, device=self.device)

#         u = pts_px[:, 0]
#         v = pts_px[:, 1]
        
#         out_h, out_w = self.cart_frame_size

#         # scale factor 
#         scale = (self.r_max - self.r_min) / out_h
         
#         # Pixels -> Physicals (meters)
#         x = (u - out_w / 2.0) * scale
#         y = (out_h - v) * scale + self.r_min
        
#         return torch.stack([x, y], dim=1)

#     def fls_filter(self, frame): # Added 'self' here just in case it was a missing instance method
#         # convert torch tensor to numpy array
#         device = frame.device 
#         _, _, c, h, w = frame.shape
#         frame_reshaped = frame.view(h, w).unsqueeze(-1)
#         frame_np = frame.detach().cpu().numpy() # Fixed missing parentheses here

#         # median filter 
#         blured = cv2.medianBlur(frame_np, ksize=5)

#         # CLAHE 
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         filtered_frame = clahe.apply(blured)

#         # restore tensor form 
#         frame_torch = torch.tensor(filtered_frame)
#         frame_reshaped = frame_torch.view(1, 1, 1, h, w)
#         frame_reshaped = frame_reshaped.to(device) # Fixed inplace move

#         return frame_reshaped