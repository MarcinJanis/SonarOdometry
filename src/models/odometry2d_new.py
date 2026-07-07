
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
                     ref_frame_orient='sim'):
            
            super().__init__()
            self.device = device 
    
            yaw_offset = sonar_config.position.yaw
            x_offset = sonar_config.position.x
            y_offset = sonar_config.position.y
            
            self.T_R_S_2d = np.array([
                [np.cos(yaw_offset), -np.sin(yaw_offset), x_offset],
                [np.sin(yaw_offset),  np.cos(yaw_offset), y_offset],
                [0,                   0,                  1]
            ])
            self.T_S_R_2d = np.linalg.inv(self.T_R_S_2d)
            
            self.ref_frame_orient = ref_frame_orient 
            self.depth_compesation = depth_compesation
            self.key_frames = key_frames 
            
            # --- Pobieranie z modelu YAML przez kropki ---
            self.use_fls_filter = model_config.filtering.use_fls_filter
            self.use_spatial_bucketing = model_config.filtering.use_spatial_bucketing
            self.use_weighted_kabsch = model_config.filtering.use_weighted_kabsch
            self.use_range_masking = model_config.filtering.use_range_masking
            
            self.max_valid_range_ratio = 0.85 
            self.bucket_grid = (4, 4) 
            self.max_pts_per_bucket = 20      
            
            # Parametry z sekcji keyframe_management
            self.key_frames_min_dist = model_config.keyframe_management.key_frames_min_dist
            self.key_frames_min_rot = model_config.keyframe_management.key_frames_min_rot
            
            # Parametry z sekcji feature_matching
            self.pts_match_thresh = model_config.feature_matching.pts_match_thresh
            self.ransac_thresh = model_config.feature_matching.ransac_thresh
            self.min_inliers_abs = model_config.feature_matching.min_inliers_abs
            self.min_inliers_ratio = model_config.feature_matching.min_inliers_ratio
                         
            self.input_img_format = input_img_format
            if self.input_img_format == 'polar':
                h = model_config.input_dimensions.polar_height
                self.cart_frame_size = (h, 2 * h)
            else:
                self.cart_frame_size = (model_config.input_dimensions.cart_height, 
                                        model_config.input_dimensions.cart_width)
        
            self.r_min = sonar_config.range.min
            self.r_max = sonar_config.range.max
            self.theta_max = sonar_config.fov.horizontal
    
            self.match_points = LoFTR(pretrained='outdoor').to(device).eval()
            
            self.window_size = 3 
            self.sliding_window = [] 
            self.current_pose = None
            self.last_frame_data = None 
            
            self.polar2cart_grid = None
            self.polar2cart_mask = None

    def fls_filter(self, frame):
        device = frame.device 
        b, c, h, w = frame.shape
        out_frames = []
        for i in range(b):
            frame_np = frame[i, 0].detach().cpu().numpy()
            frame_uint8 = np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)
            blured = cv2.medianBlur(frame_uint8, ksize=3)
            bilateral = cv2.bilateralFilter(blured, d=5, sigmaColor=25, sigmaSpace=5)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            filtered = clahe.apply(bilateral)
            filtered_float = filtered.astype(np.float32) / 255.0
            out_frames.append(torch.tensor(filtered_float, device=device).unsqueeze(0))
        return torch.stack(out_frames, dim=0)

    def set_init_state(self, init_x, init_y, init_azimuth, init_frame, init_depth, carth_mask=None):
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
        self.sliding_window = [(first_frame, self.polar2cart_mask, init_pose, init_depth)]
        self.last_frame_data = (first_frame, self.polar2cart_mask, init_pose, init_depth)

    # --- NOWA METODA: Wymuszanie dystrybucji przestrzennej dopasowań ---
    def _apply_spatial_bucketing(self, pts1, pts2, conf):
        out_h, out_w = self.cart_frame_size
        grid_rows, grid_cols = self.bucket_grid
        
        dy = out_h / grid_rows
        dx = out_w / grid_cols
        
        pts1_np = pts1.cpu().numpy()
        buckets = {}
        
        for i, (pt, c) in enumerate(zip(pts1_np, conf)):
            r_idx = int(pt[1] // dy)
            c_idx = int(pt[0] // dx)
            # Ograniczenie indeksów, by uniknąć błędu brzegowego
            r_idx = min(max(r_idx, 0), grid_rows - 1)
            c_idx = min(max(c_idx, 0), grid_cols - 1)
            
            bucket_key = (r_idx, c_idx)
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append((i, c.item()))
            
        selected_indices = []
        for b_key, items in buckets.items():
            # Sortowanie malejąco po ufności wewnątrz bucketa
            items.sort(key=lambda x: x[1], reverse=True)
            # Wybieramy tylko N najsilniejszych punktów
            selected_indices.extend([x[0] for x in items[:self.max_pts_per_bucket]])
            
        return pts1[selected_indices], pts2[selected_indices], conf[selected_indices]

    # --- NOWA METODA: Algorytm Ważonych Najmniejszych Kwadratów SVD ---
    def _weighted_kabsch(self, pts_ref, pts_curr, weights):
        """ Rozwiązuje problem estymacji transformacji sztywnej (R, t) ważąc punkty poprzez ich pewność """
        W = weights / weights.sum()
        
        # Środki ciężkości chmur (uwzględniające wagi)
        centroid_ref = np.sum(pts_ref * W[:, np.newaxis], axis=0)
        centroid_curr = np.sum(pts_curr * W[:, np.newaxis], axis=0)
        
        # Centrowanie punktów
        p_ref = pts_ref - centroid_ref
        p_curr = pts_curr - centroid_curr
        
        # Macierz kowariancji z wagami (H = P_curr^T * W * P_ref)
        H = (p_curr.T * W) @ p_ref
        
        # Rozkład SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ochrona przed odbiciem lustrzanym (det(R) musi wynosić 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            
        t = centroid_ref - R @ centroid_curr
        return R, t

    # def _match_and_estimate(self, ref_frame, ref_mask, ref_pose, ref_depth, new_frame, new_depth):
    #     matches = self.match_points({'image0': ref_frame, 'mask0': ref_mask, 'image1': new_frame, 'mask1': self.polar2cart_mask})
    #     pts1, pts2, confidence = matches['keypoints0'], matches['keypoints1'], matches['confidence']
        
    #     valid_matches = confidence > self.pts_match_thresh
    #     pts1, pts2, confidence = pts1[valid_matches], pts2[valid_matches], confidence[valid_matches]
        
    #     if len(pts1) < 3: return None 

    #     # 1. SPATIAL BUCKETING
    #     if self.use_spatial_bucketing:
    #         pts1, pts2, confidence = self._apply_spatial_bucketing(pts1, pts2, confidence)

    #     pts1_r, pts2_r = self.scale_px2physcial(pts1), self.scale_px2physcial(pts2)
        
    #     # 2. KOMPENSACJA DEPTH & MASKA ZASIĘGU
    #     ray1 = torch.sqrt(pts1_r[:, 0]**2 + pts1_r[:, 1]**2)
    #     ray2 = torch.sqrt(pts2_r[:, 0]**2 + pts2_r[:, 1]**2)
        
    #     valid_mask = torch.ones(len(ray1), dtype=torch.bool, device=self.device)
        
    #     if self.depth_compesation:
    #         valid_mask = valid_mask & (ray1 > ref_depth) & (ray2 > new_depth)
            
    #     if self.use_range_masking:
    #         valid_mask = valid_mask & (ray1 < self.r_max * self.max_valid_range_ratio) & (ray2 < self.r_max * self.max_valid_range_ratio)
            
    #     pts1_r, pts2_r = pts1_r[valid_mask], pts2_r[valid_mask]
    #     ray1, ray2 = ray1[valid_mask], ray2[valid_mask]
    #     confidence = confidence[valid_mask]
        
    #     if self.depth_compesation:
    #         scale1 = torch.sqrt(ray1**2 - ref_depth**2) / ray1
    #         scale2 = torch.sqrt(ray2**2 - new_depth**2) / ray2
    #         pts1_r = pts1_r * scale1.unsqueeze(1)
    #         pts2_r = pts2_r * scale2.unsqueeze(1)

    #     pts1_np, pts2_np = pts1_r.cpu().numpy(), pts2_r.cpu().numpy()
    #     conf_np = confidence.cpu().numpy()
        
    #     if len(pts1_np) < 3: return None

    #     # Standardowy RANSAC (wykorzystujemy go TYLKO do odfiltrowania drastycznych outlierów)
    #     M, inlier_mask = cv2.estimateAffinePartial2D(
    #         pts2_np, pts1_np, method=cv2.RANSAC, 
    #         ransacReprojThreshold=self.ransac_thresh, maxIters=3000, confidence=0.999
    #     )

    #     if M is not None and inlier_mask is not None:
    #         inlier_mask = inlier_mask.ravel().astype(bool)
    #         inliers_abs = int(inlier_mask.sum())
    #         inliers_p = inliers_abs / len(pts1_np) if len(pts1_np) > 0 else 0.0
            
    #         if inliers_abs >= self.min_inliers_abs and inliers_p >= self.min_inliers_ratio:
                
    #             # 3. WAŻONY ALGORYTM KABSCHA (Zamiast nieprecyzyjnej macierzy z OpenCV)
    #             if self.use_weighted_kabsch:
    #                 pts1_inliers = pts1_np[inlier_mask]
    #                 pts2_inliers = pts2_np[inlier_mask]
    #                 conf_inliers = conf_np[inlier_mask]
                    
    #                 R_opt, t_opt = self._weighted_kabsch(pts1_inliers, pts2_inliers, conf_inliers)
    #                 angle = np.arctan2(R_opt[1, 0], R_opt[0, 0])
    #                 raw_tx_sonar, raw_ty_sonar = float(t_opt[0]), float(t_opt[1])
    #             else:
    #                 # Klasyczny odczyt z RANSACa dla porównania (jeśli używamy flagi False)
    #                 angle = np.arctan2(M[1, 0], M[0, 0])
    #                 raw_tx_sonar, raw_ty_sonar = float(M[0, 2]), float(M[1, 2])
                
    #             theta = -angle if self.ref_frame_orient == 'sim' else angle
    #             tx, ty = (raw_ty_sonar, -raw_tx_sonar) if self.ref_frame_orient == 'sim' else (-raw_ty_sonar, -raw_tx_sonar)
                
    #             local_T = np.array([[np.cos(theta), -np.sin(theta), tx], 
    #                                 [np.sin(theta), np.cos(theta), ty], 
    #                                 [0, 0, 1]])
                
    #             est_pose = ref_pose @ (self.T_R_S_2d @ local_T @ self.T_S_R_2d)
                
    #             return {
    #                 'est_pose': est_pose, 'pts1': pts1[inlier_mask], 'pts2': pts2[inlier_mask], 'confidence': confidence[inlier_mask],
    #                 'inliers_abs': inliers_abs, 'inliers_p': inliers_p, 'ref_frame': ref_frame,
    #                 'raw_tx_sonar': raw_tx_sonar, 'raw_ty_sonar': raw_ty_sonar
    #             }
    #     return None
    def _match_and_estimate(self, ref_frame, ref_mask, ref_pose, ref_depth, new_frame, new_depth):
        matches = self.match_points({'image0': ref_frame, 'mask0': ref_mask, 'image1': new_frame, 'mask1': self.polar2cart_mask})
        pts1, pts2, confidence = matches['keypoints0'], matches['keypoints1'], matches['confidence']
        
        valid_matches = confidence > self.pts_match_thresh
        pts1, pts2, confidence = pts1[valid_matches], pts2[valid_matches], confidence[valid_matches]
        
        if len(pts1) < 3: return None 

        # 1. SPATIAL BUCKETING
        if self.use_spatial_bucketing:
            pts1, pts2, confidence = self._apply_spatial_bucketing(pts1, pts2, confidence)

        pts1_r, pts2_r = self.scale_px2physcial(pts1), self.scale_px2physcial(pts2)
        
        # 2. KOMPENSACJA DEPTH & MASKA ZASIĘGU
        ray1 = torch.sqrt(pts1_r[:, 0]**2 + pts1_r[:, 1]**2)
        ray2 = torch.sqrt(pts2_r[:, 0]**2 + pts2_r[:, 1]**2)
        
        valid_mask = torch.ones(len(ray1), dtype=torch.bool, device=self.device)
        
        if self.depth_compesation:
            valid_mask = valid_mask & (ray1 > ref_depth) & (ray2 > new_depth)
            
        if self.use_range_masking:
            valid_mask = valid_mask & (ray1 < self.r_max * self.max_valid_range_ratio) & (ray2 < self.r_max * self.max_valid_range_ratio)
            
        # ZASTOSOWANIE MASKI DO WSZYSTKICH WEKTORÓW (TUTAJ BYŁ BŁĄD)
        pts1_r, pts2_r = pts1_r[valid_mask], pts2_r[valid_mask]
        pts1, pts2 = pts1[valid_mask], pts2[valid_mask] # <--- DODANA LINIA
        ray1, ray2 = ray1[valid_mask], ray2[valid_mask]
        confidence = confidence[valid_mask]
        
        if self.depth_compesation:
            scale1 = torch.sqrt(ray1**2 - ref_depth**2) / ray1
            scale2 = torch.sqrt(ray2**2 - new_depth**2) / ray2
            pts1_r = pts1_r * scale1.unsqueeze(1)
            pts2_r = pts2_r * scale2.unsqueeze(1)

        pts1_np, pts2_np = pts1_r.cpu().numpy(), pts2_r.cpu().numpy()
        conf_np = confidence.cpu().numpy()
        
        if len(pts1_np) < 3: return None

        # Standardowy RANSAC (wykorzystujemy go TYLKO do odfiltrowania drastycznych outlierów)
        M, inlier_mask = cv2.estimateAffinePartial2D(
            pts2_np, pts1_np, method=cv2.RANSAC, 
            ransacReprojThreshold=self.ransac_thresh, maxIters=3000, confidence=0.999
        )

        if M is not None and inlier_mask is not None:
            inlier_mask = inlier_mask.ravel().astype(bool)
            inliers_abs = int(inlier_mask.sum())
            inliers_p = inliers_abs / len(pts1_np) if len(pts1_np) > 0 else 0.0
            
            if inliers_abs >= self.min_inliers_abs and inliers_p >= self.min_inliers_ratio:
                
                # 3. WAŻONY ALGORYTM KABSCHA (Zamiast nieprecyzyjnej macierzy z OpenCV)
                if self.use_weighted_kabsch:
                    pts1_inliers = pts1_np[inlier_mask]
                    pts2_inliers = pts2_np[inlier_mask]
                    conf_inliers = conf_np[inlier_mask]
                    
                    R_opt, t_opt = self._weighted_kabsch(pts1_inliers, pts2_inliers, conf_inliers)
                    angle = np.arctan2(R_opt[1, 0], R_opt[0, 0])
                    raw_tx_sonar, raw_ty_sonar = float(t_opt[0]), float(t_opt[1])
                else:
                    # Klasyczny odczyt z RANSACa dla porównania (jeśli używamy flagi False)
                    angle = np.arctan2(M[1, 0], M[0, 0])
                    raw_tx_sonar, raw_ty_sonar = float(M[0, 2]), float(M[1, 2])
                
                theta = -angle if self.ref_frame_orient == 'sim' else angle
                tx, ty = (raw_ty_sonar, raw_tx_sonar) if self.ref_frame_orient == 'sim' else (-raw_ty_sonar, -raw_tx_sonar)
                
                local_T = np.array([[np.cos(theta), -np.sin(theta), tx], 
                                    [np.sin(theta), np.cos(theta), ty], 
                                    [0, 0, 1]])
                
                est_pose = ref_pose @ (self.T_R_S_2d @ local_T @ self.T_S_R_2d)
                
                return {
                    'est_pose': est_pose, 'pts1': pts1[inlier_mask], 'pts2': pts2[inlier_mask], 'confidence': confidence[inlier_mask],
                    'inliers_abs': inliers_abs, 'inliers_p': inliers_p, 'ref_frame': ref_frame,
                    'raw_tx_sonar': raw_tx_sonar, 'raw_ty_sonar': raw_ty_sonar
                }
        return None


    @torch.no_grad()
    def forward(self, frame, depth, return_visu=False):
        if self.use_fls_filter:
            frame = self.fls_filter(frame)

        new_frame = self.polar2car(frame) if self.input_img_format == 'polar' else frame
        
        est_poses = []
        latest_visu_match = None 

        for i, (ref_frame, ref_mask, ref_pose, ref_depth) in enumerate(self.sliding_window):
            match_res = self._match_and_estimate(ref_frame, ref_mask, ref_pose, ref_depth, new_frame, depth)
            if match_res is not None:
                est_poses.append(match_res)
                if i == len(self.sliding_window) - 1:
                    latest_visu_match = match_res

        if len(est_poses) == 0 and self.last_frame_data is not None:
            ref_frame, ref_mask, ref_pose, ref_depth = self.last_frame_data
            match_res = self._match_and_estimate(ref_frame, ref_mask, ref_pose, ref_depth, new_frame, depth)
            if match_res is not None:
                est_poses.append(match_res)
                latest_visu_match = match_res

        step_is_valid = False
        if len(est_poses) > 0:
            est_x_list = [p['est_pose'][0, 2] for p in est_poses]
            est_y_list = [p['est_pose'][1, 2] for p in est_poses]
            est_yaw_list = [np.arctan2(p['est_pose'][1, 0], p['est_pose'][0, 0]) for p in est_poses]
            
            median_x, median_y = np.median(est_x_list), np.median(est_y_list)
            median_azimuth = np.arctan2(np.sum(np.sin(est_yaw_list)), np.sum(np.cos(est_yaw_list)))
            
            # --- Zastosowanie Tłumienia Poprzecznego (TY DAMPING) ---
            # Liczymy deltę pozy względem ostatniej AKCEPTOWANEJ ramki w układzie lokalnym robota
            R_curr = self.current_pose[0:2, 0:2]
            t_curr = self.current_pose[0:2, 2]
            
            t_new_raw = np.array([median_x, median_y])
            delta_t_local = R_curr.T @ (t_new_raw - t_curr)
            
            # Utwardzamy trajektorię
            ty_damping_factor = 1.0 # <- Możesz regulować (np. 0.3 dławi mocniej, 1.0 wyłącza)
            delta_t_local_damped = np.array([delta_t_local[0], delta_t_local[1] * ty_damping_factor])
            
            t_new_damped = t_curr + R_curr @ delta_t_local_damped
            
            new_pose = np.array([[np.cos(median_azimuth), -np.sin(median_azimuth), t_new_damped[0]], 
                                 [np.sin(median_azimuth), np.cos(median_azimuth), t_new_damped[1]], 
                                 [0, 0, 1]])
            step_is_valid = True
        else:
            new_pose = self.current_pose

        global_x, global_y = float(new_pose[0, 2]), float(new_pose[1, 2])
        global_azimuth = float(np.arctan2(new_pose[1, 0], new_pose[0, 0]))

        _, _, latest_kf_pose, _ = self.sliding_window[-1]
        dist = np.sqrt((global_x - latest_kf_pose[0, 2])**2 + (global_y - latest_kf_pose[1, 2])**2)
        
        prev_azimuth = np.arctan2(latest_kf_pose[1, 0], latest_kf_pose[0, 0])
        azimuth_diff = np.abs(np.arctan2(np.sin(global_azimuth - prev_azimuth), np.cos(global_azimuth - prev_azimuth)))
        
        key_frame_detected = step_is_valid and (dist >= self.key_frames_min_dist or azimuth_diff >= self.key_frames_min_rot)
        
        if key_frame_detected:
            self.sliding_window.append((new_frame, self.polar2cart_mask, new_pose, depth))
            if len(self.sliding_window) > self.window_size: 
                self.sliding_window.pop(0)

        self.last_frame_data = (new_frame, self.polar2cart_mask, new_pose, depth)
        self.current_pose = new_pose

        if not return_visu:
            return (global_x, global_y), global_azimuth
        else:
            R_kf, t_kf = latest_kf_pose[0:2, 0:2], latest_kf_pose[0:2, 2]
            R_new, t_new = new_pose[0:2, 0:2], new_pose[0:2, 2]

            R_rel = R_kf.T @ R_new
            t_rel = R_kf.T @ (t_new - t_kf)

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
            
            individual_estimates = list(zip(
                [p['est_pose'][0, 2] for p in est_poses], 
                [p['est_pose'][1, 2] for p in est_poses], 
                [np.arctan2(p['est_pose'][1, 0], p['est_pose'][0, 0]) for p in est_poses]
            )) if len(est_poses) > 0 else []

            visu = {
                'combined_imgs': frames_np_rgb,
                'pts1': pts1_visu, 'pts2': pts2_visu, 'pts2_offset': (0, w),
                'inliers_ratio': v_inliers_ratio, 'inliers_abs': v_inliers_abs,
                'matches_total': len(pts1_visu), 'mean_matched_confidence': conf_visu,
                'key_frame_detected': key_frame_detected, 'step_is_valid': step_is_valid,
                'tx_sonar': float(v_raw_tx_sonar), 'ty_sonar': float(v_raw_ty_sonar),
                'tx_mapped': float(t_rel[0]), 'ty_mapped': float(t_rel[1]), 'theta': float(np.arctan2(R_rel[1, 0], R_rel[0, 0])),
                'displacement': float(dist), 'azimuth_diff': float(azimuth_diff),
                'global_pose': (global_x, global_y, global_azimuth),
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
    
    
# import torch
# import torch.nn.functional as F
# import torch.nn as nn 
# from kornia.feature import LoFTR

# import numpy as np 
# import cv2 

# class sonar_odometry(nn.Module):

#     def __init__(self, model_config, sonar_config, device, 
#                  depth_compesation=True,
#                  key_frames=True,
#                  input_img_format='polar',
#                  ref_frame_orient='sim', # 'sim', 'aracati'
#                  use_fls_filter=True):
        
#         super().__init__()
#         self.device = device 

#         # --- Kompensacja Dźwigni (Extrinsics) ---
#         yaw_offset = sonar_config.position.yaw
#         x_offset = sonar_config.position.x
#         y_offset = sonar_config.position.y
        
#         self.T_R_S_2d = np.array([
#             [np.cos(yaw_offset), -np.sin(yaw_offset), x_offset],
#             [np.sin(yaw_offset),  np.cos(yaw_offset), y_offset],
#             [0,                   0,                  1]
#         ])
#         self.T_S_R_2d = np.linalg.inv(self.T_R_S_2d)
        
#         # --- Konfiguracja Systemu ---
#         self.ref_frame_orient = ref_frame_orient 
#         self.depth_compesation = depth_compesation
#         self.key_frames = key_frames 
#         self.use_fls_filter = getattr(model_config, 'use_fls_filter', use_fls_filter)
        
#         # Parametry logiczne
#         self.key_frames_min_dist = model_config.key_frames_min_dist
#         self.key_frames_min_rot = model_config.key_frames_min_rot
#         self.pts_match_thresh = model_config.pts_match_thresh
#         self.ransac_thresh = model_config.ransac_thresh
        
#         self.min_inliers_abs = getattr(model_config, 'min_inliers_abs', 12)
#         self.min_inliers_ratio = getattr(model_config, 'min_inliers_ratio', 0.06)
                     
#         # Sonar configuration
#         self.input_img_format = input_img_format
#         if self.input_img_format == 'polar':
#             self.cart_frame_size = (model_config.POLAR_FLS_INPUT_HEIGHT, 2 * model_config.POLAR_FLS_INPUT_HEIGHT)
#         else:
#             self.cart_frame_size = (model_config.CART_FLS_INPUT_HEIGHT, model_config.CART_FLS_INPUT_WIDTH)
    
#         self.r_min = sonar_config.range.min
#         self.r_max = sonar_config.range.max
#         self.theta_max = sonar_config.fov.horizontal

#         self.match_points = LoFTR(pretrained='outdoor').to(device).eval()
        
#         # Stan estymatora (Window Size + Cache)
#         self.window_size = 3
#         self.sliding_window = [] # Zapisuje tuple: (frame, mask, pose, depth)
#         self.current_pose = None
#         self.last_frame_data = None 
        
#         self.polar2cart_grid = None
#         self.polar2cart_mask = None


#         self.side_drift_damping = 0.05

#     def fls_filter(self, frame):
#         device = frame.device 
#         b, c, h, w = frame.shape
#         out_frames = []
#         for i in range(b):
#             frame_np = frame[i, 0].detach().cpu().numpy()
#             frame_uint8 = np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)
            
#             blured = cv2.medianBlur(frame_uint8, ksize=3)
#             bilateral = cv2.bilateralFilter(blured, d=5, sigmaColor=25, sigmaSpace=5)
#             clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
#             filtered = clahe.apply(bilateral)
            
#             filtered_float = filtered.astype(np.float32) / 255.0
#             out_frames.append(torch.tensor(filtered_float, device=device).unsqueeze(0))
#         return torch.stack(out_frames, dim=0)

#     def set_init_state(self, init_x, init_y, init_azimuth, init_frame, init_depth, carth_mask=None):
#         """ Dodano parametr `init_depth` by móc skompensować historyczne punkty klatki początkowej """
#         if self.use_fls_filter:
#             init_frame = self.fls_filter(init_frame)

#         b, c, h, w = init_frame.shape
#         out_h, out_w = h, 2 * h

#         self.cart_frame_size = (out_h, out_w)

#         y = torch.arange(out_h, device=self.device, dtype=torch.float32)
#         x = torch.arange(out_w, device=self.device, dtype=torch.float32)
#         y, x = torch.meshgrid(y, x, indexing='ij')
#         x = x - out_w / 2.0
#         y = out_h - y

#         scale = (self.r_max - self.r_min) / out_h
#         x_r = x * scale
#         y_r = y * scale + self.r_min
#         r = torch.sqrt(x_r**2 + y_r**2)
#         theta = torch.atan2(x_r, torch.clamp(y_r, min=1e-5))
        
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

#         init_pose = np.array([[np.cos(init_azimuth), -np.sin(init_azimuth), init_x], 
#                               [np.sin(init_azimuth),  np.cos(init_azimuth), init_y], 
#                               [0,                     0,                    1]])
        
#         first_frame = self.polar2car(init_frame)
#         self.current_pose = init_pose
        
#         # Bufor przechowuje teraz depth!
#         self.sliding_window = [(first_frame, self.polar2cart_mask, init_pose, init_depth)]
#         self.last_frame_data = (first_frame, self.polar2cart_mask, init_pose, init_depth)

#     def _match_and_estimate(self, ref_frame, ref_mask, ref_pose, ref_depth, new_frame, new_depth):
#         matches = self.match_points({'image0': ref_frame, 'mask0': ref_mask, 'image1': new_frame, 'mask1': self.polar2cart_mask})
#         pts1, pts2, confidence = matches['keypoints0'], matches['keypoints1'], matches['confidence']
        
#         valid_matches = confidence > self.pts_match_thresh
#         pts1, pts2 = pts1[valid_matches], pts2[valid_matches]
        
#         if len(pts1) < 3: return None 

#         pts1_r, pts2_r = self.scale_px2physcial(pts1), self.scale_px2physcial(pts2)
        
#         # --- POPRAWIONA LOGIKA DEPTH COMPENSATION ---
#         if self.depth_compesation:
#             ray1 = torch.sqrt(pts1_r[:, 0]**2 + pts1_r[:, 1]**2)
#             ray2 = torch.sqrt(pts2_r[:, 0]**2 + pts2_r[:, 1]**2)
            
#             # Weryfikujemy punkty w oparciu o głębokość zmierzoną w odpowiedniej chwili czasowej
#             valid_mask = (ray1 > ref_depth) & (ray2 > new_depth)
            
#             pts1_r, pts2_r = pts1_r[valid_mask], pts2_r[valid_mask]
#             ray1, ray2 = ray1[valid_mask], ray2[valid_mask]
            
#             # Prawidłowe niezależne skalowanie
#             scale1 = torch.sqrt(ray1**2 - ref_depth**2) / ray1
#             scale2 = torch.sqrt(ray2**2 - new_depth**2) / ray2
            
#             pts1_r = pts1_r * scale1.unsqueeze(1)
#             pts2_r = pts2_r * scale2.unsqueeze(1)
#         # ---------------------------------------------

#         pts1_np, pts2_np = pts1_r.cpu().numpy(), pts2_r.cpu().numpy()
#         if len(pts1_np) < 3: return None

#         M, inlier_mask = cv2.estimateAffinePartial2D(
#             pts2_np, pts1_np, method=cv2.RANSAC, 
#             ransacReprojThreshold=self.ransac_thresh, maxIters=3000, confidence=0.999
#         )

#         if M is not None and inlier_mask is not None:
#             inlier_mask = inlier_mask.ravel().astype(bool)
#             inliers_abs = int(inlier_mask.sum())
#             inliers_p = inliers_abs / len(pts1_np) if len(pts1_np) > 0 else 0.0
            
#             if inliers_abs >= self.min_inliers_abs and inliers_p >= self.min_inliers_ratio:
#                 # Odczyt czystej macierzy z OpenCV
#                 angle = np.arctan2(M[1, 0], M[0, 0])
#                 raw_tx_sonar = float(M[0, 2])
#                 raw_ty_sonar = float(M[1, 2])
                
#                 theta = -angle if self.ref_frame_orient == 'sim' else angle
#                 tx, ty = (raw_ty_sonar, -raw_tx_sonar) if self.ref_frame_orient == 'sim' else (-raw_ty_sonar, -raw_tx_sonar)
                
        
               
#                 ty = ty * self.side_drift_damping

#                 local_T = np.array([[np.cos(theta), -np.sin(theta), tx], 
#                                     [np.sin(theta), np.cos(theta), ty], 
#                                     [0, 0, 1]])
                
#                 est_pose = ref_pose @ (self.T_R_S_2d @ local_T @ self.T_S_R_2d)
                
#                 return {
#                     'est_pose': est_pose, 'pts1': pts1, 'pts2': pts2, 'confidence': confidence,
#                     'inliers_abs': inliers_abs, 'inliers_p': inliers_p, 'ref_frame': ref_frame,
#                     'raw_tx_sonar': raw_tx_sonar, 'raw_ty_sonar': raw_ty_sonar
#                 }
#         return None

#     @torch.no_grad()
#     def forward(self, frame, depth, return_visu=False):
#         if self.use_fls_filter:
#             frame = self.fls_filter(frame)

#         new_frame = self.polar2car(frame) if self.input_img_format == 'polar' else frame
        
#         est_poses = []
#         latest_visu_match = None 

#         # 1. Okno Multi-Frame Tracking
#         for i, (ref_frame, ref_mask, ref_pose, ref_depth) in enumerate(self.sliding_window):
#             match_res = self._match_and_estimate(ref_frame, ref_mask, ref_pose, ref_depth, new_frame, depth)
#             if match_res is not None:
#                 est_poses.append(match_res)
#                 if i == len(self.sliding_window) - 1:
#                     latest_visu_match = match_res

#         # 2. Awaryjny odczyt względem poprzedniej klatki (Jeśli okno nawaliło)
#         if len(est_poses) == 0 and self.last_frame_data is not None:
#             ref_frame, ref_mask, ref_pose, ref_depth = self.last_frame_data
#             match_res = self._match_and_estimate(ref_frame, ref_mask, ref_pose, ref_depth, new_frame, depth)
#             if match_res is not None:
#                 est_poses.append(match_res)
#                 latest_visu_match = match_res

#         # 3. Agregacja i estymacja czystej pozy 
#         step_is_valid = False
#         if len(est_poses) > 0:
#             est_x_list = [p['est_pose'][0, 2] for p in est_poses]
#             est_y_list = [p['est_pose'][1, 2] for p in est_poses]
#             est_yaw_list = [np.arctan2(p['est_pose'][1, 0], p['est_pose'][0, 0]) for p in est_poses]
            
#             median_x, median_y = np.median(est_x_list), np.median(est_y_list)
#             median_azimuth = np.arctan2(np.sum(np.sin(est_yaw_list)), np.sum(np.cos(est_yaw_list)))
            
#             new_pose = np.array([[np.cos(median_azimuth), -np.sin(median_azimuth), median_x], 
#                                  [np.sin(median_azimuth), np.cos(median_azimuth), median_y], 
#                                  [0, 0, 1]])
#             step_is_valid = True
#         else:
#             # Fallback zero-velocity: stoisz w miejscu
#             new_pose = self.current_pose

#         global_x, global_y = float(new_pose[0, 2]), float(new_pose[1, 2])
#         global_azimuth = float(np.arctan2(new_pose[1, 0], new_pose[0, 0]))

#         # 4. Logika Klatek Kluczowych (Keyframes)
#         _, _, latest_kf_pose, _ = self.sliding_window[-1]
#         dist = np.sqrt((global_x - latest_kf_pose[0, 2])**2 + (global_y - latest_kf_pose[1, 2])**2)
        
#         prev_azimuth = np.arctan2(latest_kf_pose[1, 0], latest_kf_pose[0, 0])
#         azimuth_diff = np.abs(np.arctan2(np.sin(global_azimuth - prev_azimuth), np.cos(global_azimuth - prev_azimuth)))
        
#         key_frame_detected = step_is_valid and (dist >= self.key_frames_min_dist or azimuth_diff >= self.key_frames_min_rot)
        
#         if key_frame_detected:
#             # Dodano zapisywanie depth jako czwartego elementu tupli w oknie przesuwnym
#             self.sliding_window.append((new_frame, self.polar2cart_mask, new_pose, depth))
#             if len(self.sliding_window) > self.window_size: 
#                 self.sliding_window.pop(0)

#         # Cache'owanie zawsze bieżącej klatki (na potrzeby fallbacku)
#         self.last_frame_data = (new_frame, self.polar2cart_mask, new_pose, depth)
#         self.current_pose = new_pose

#         if not return_visu:
#             return (global_x, global_y), global_azimuth
#         else:
#             # Obliczenia diagnostyczne dla pętli głównej
#             R_kf, t_kf = latest_kf_pose[0:2, 0:2], latest_kf_pose[0:2, 2]
#             R_new, t_new = new_pose[0:2, 0:2], new_pose[0:2, 2]

#             R_rel = R_kf.T @ R_new
#             t_rel = R_kf.T @ (t_new - t_kf)

#             b, c, h, w = new_frame.shape
#             if latest_visu_match is not None:
#                 frame1_np = latest_visu_match['ref_frame'].squeeze(0).permute(1, 2, 0).cpu().numpy()
#                 pts1_visu = latest_visu_match['pts1'].cpu().numpy()
#                 pts2_visu = latest_visu_match['pts2'].cpu().numpy()
#                 conf_visu = float(latest_visu_match['confidence'].mean().cpu().numpy())
#                 v_inliers_abs = latest_visu_match['inliers_abs']
#                 v_inliers_ratio = latest_visu_match['inliers_p']
#                 v_raw_tx_sonar = latest_visu_match['raw_tx_sonar']
#                 v_raw_ty_sonar = latest_visu_match['raw_ty_sonar']
#             else:
#                 frame1_np = self.sliding_window[-1][0].squeeze(0).permute(1, 2, 0).cpu().numpy()
#                 pts1_visu, pts2_visu = np.zeros((0,2)), np.zeros((0,2))
#                 conf_visu, v_inliers_abs, v_inliers_ratio = 0.0, 0, 0.0
#                 v_raw_tx_sonar, v_raw_ty_sonar = 0.0, 0.0

#             frame2_np = new_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             combined_img_gray = np.concatenate((frame1_np, frame2_np), axis=1)
#             frames_np_rgb = cv2.cvtColor(combined_img_gray, cv2.COLOR_GRAY2RGB) if combined_img_gray.shape[-1] == 1 else combined_img_gray
            
#             individual_estimates = list(zip(
#                 [p['est_pose'][0, 2] for p in est_poses], 
#                 [p['est_pose'][1, 2] for p in est_poses], 
#                 [np.arctan2(p['est_pose'][1, 0], p['est_pose'][0, 0]) for p in est_poses]
#             )) if len(est_poses) > 0 else []

#             visu = {
#                 'combined_imgs': frames_np_rgb,
#                 'pts1': pts1_visu, 'pts2': pts2_visu, 'pts2_offset': (0, w),
#                 'inliers_ratio': v_inliers_ratio, 'inliers_abs': v_inliers_abs,
#                 'matches_total': len(pts1_visu), 'mean_matched_confidence': conf_visu,
#                 'key_frame_detected': key_frame_detected, 'step_is_valid': step_is_valid,
#                 'tx_sonar': float(v_raw_tx_sonar), 'ty_sonar': float(v_raw_ty_sonar),
#                 'tx_mapped': float(t_rel[0]), 'ty_mapped': float(t_rel[1]), 'theta': float(np.arctan2(R_rel[1, 0], R_rel[0, 0])),
#                 'displacement': float(dist), 'azimuth_diff': float(azimuth_diff),
#                 'global_pose': (global_x, global_y, global_azimuth),
#                 'window_matches_count': f"{len(est_poses)}/{len(self.sliding_window)}",
#                 'individual_estimates': individual_estimates
#             }
#             return (global_x, global_y), global_azimuth, visu

#     def polar2car(self, frame):
#         out = F.grid_sample(frame, self.polar2cart_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
#         return out * self.polar2cart_mask.unsqueeze(1)

#     def scale_px2physcial(self, pts_px):
#         out_h, out_w = self.cart_frame_size
#         scale = (self.r_max - self.r_min) / out_h
#         x = (pts_px[:, 0] - out_w / 2.0) * scale
#         y = (out_h - pts_px[:, 1]) * scale + self.r_min
#         return torch.stack([x, y], dim=1)
