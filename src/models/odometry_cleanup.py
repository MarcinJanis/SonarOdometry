import torch
import torch.nn.functional as F
import torch.nn as nn 
from kornia.feature import LoFTR

import numpy as np 
import cv2 

class sonar_odometry(nn.Module):
  
    def __init__(self, model_config, sonar_config, 
                     device, 
                     depth_compesation=True,
                     key_frames=True,
                     input_img_format='polar',
                     ref_frame_orient='sim'):
            
            super().__init__()
                       
            self.device = device 

            # --- Configuration ---

            # extrinsic calib 
            yaw_offset = sonar_config.position.yaw
            x_offset = sonar_config.position.x
            y_offset = sonar_config.position.y

            self.T_R_S_2d = np.array([
                [np.cos(yaw_offset), -np.sin(yaw_offset), x_offset],
                [np.sin(yaw_offset),  np.cos(yaw_offset), y_offset],
                [0,                   0,                  1]
            ]) # Transform matrix Robot -> Sonar frame
            
            self.T_S_R_2d = np.linalg.inv(self.T_R_S_2d) # Transform matrix Sonar -> Robot frame

            # frame of reference orientation - define how local frame is rotated relative to global frame
            self.ref_frame_orient = ref_frame_orient # 'sim' (for own dataset) or 'aracati' 

            # activate depth compesation - only if depth data available
            self.depth_compesation = depth_compesation

            # key frame detection - based on distance from previous
            self.key_frames = key_frames 

            # activae filtration 
            self.use_fls_filter = model_config.filtering.use_fls_filter

            # spatial bucketing for equally distributed key points
            self.use_spatial_bucketing = model_config.filtering.use_spatial_bucketing

            # kabsh motion model 
            self.use_weighted_kabsch = model_config.filtering.use_weighted_kabsch

            # ????
            self.use_range_masking = model_config.filtering.use_range_masking

            # --- Parameters ---
            self.max_valid_range_ratio = 0.85 
            self.bucket_grid = (4, 4) 
            self.max_pts_per_bucket = 20      
            
            # Parametry z sekcji keyframe_management
            self.key_frames_min_dist = model_config.keyframe_management.key_frames_min_dist
            self.key_frames_min_rot = model_config.keyframe_management.key_frames_min_rot
            self.key_frame_timeout = model_config.keyframe_management.max_skip_frames
            
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
            
            self.skipped_frames = 0 

            self.polar2cart_grid = None
            self.polar2cart_mask = None
