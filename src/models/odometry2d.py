import torch
import torch.nn.functional as F
import torch.nn as nn 
from kornia.feature import LoFTR

import numpy as np 
import cv2 

from box import Box
import yaml

from .utils import project_points, approx_movement, depth_to_elev_angle, add_noise, ExtrinsicsCalib


class sonar_odometry(nn.Module):

    def __init__(self, model_config, sonar_config, device):
        super().__init__()

        self.device = device 

        # with open(model_cfg, "r") as f:
        #     model_config = Box(yaml.safe_load(f))

        # with open(sonar_cfg, "r") as f:
        #     sonar_config = Box(yaml.safe_load(f))   

        self.calib = ExtrinsicsCalib(T = [sonar_config.position.x, sonar_config.position.y, sonar_config.position.z],
                                     R = [sonar_config.position.roll, sonar_config.position.pitch, sonar_config.position.yaw])

        # --- init parameters ---
        self.pts_match_thresh = model_config.pts_match_thresh # [-]
        self.ransac_thresh = model_config.ransac_thresh # [m]

        self.polar_frame_size = (model_config.FLS_INPUT_HEIGHT, model_config.FLS_INPUT_WIDTH)
        self.cart_frame_size = (model_config.FLS_INPUT_HEIGHT, 2*model_config.FLS_INPUT_HEIGHT)


        self.r_min = sonar_config.range.min
        self.r_max = sonar_config.range.max
        self.theta_max = sonar_config.fov.horizontal

        # --- init modules ---
        pretrained = 'outdoor'
        self.match_points = LoFTR(pretrained=pretrained).to(device).eval()

        # --- init inner state --- 
        self.prev_pose= None
        self.prev_frame = None
        self.mask = None

        self.polar2cart_grid = None
        

    def set_init_state(self, init_x, init_y, init_frame):

        # --- generate sampling grid once to speed up ---
        b, c, h, w = init_frame.shape
        
        # set output shape
        out_h = h
        out_w = 2 * h

        # Inverse remapping 
        y = torch.arange(out_h, device=init_frame.device, dtype=torch.float32)
        x = torch.arange(out_w, device=init_frame.device, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing='ij')

        # Recenter
        x = x - out_w / 2.0
        y = out_h - y

        # Rescale to real-world values (metry)
        scale = (self.r_max - self.r_min) / out_h
        x_r = x * scale
        y_r = y * scale + self.r_min

        # Map (x, y) -> (theta, r)
        r = torch.sqrt(x_r**2 + y_r**2)
        y_r_clamp = torch.clamp(y_r, min=1e-5)
        theta = torch.atan2(x_r, y_r_clamp)

        # Nornalization 
        norm_theta = theta / (self.theta_max / 2.0)
        norm_r = (r - self.r_min) / (self.r_max - self.r_min) * 2.0 - 1.0

        # Crate grid with shape (b, out_h, out_w, 2)

        grid = torch.stack((norm_r, norm_theta), dim=-1).unsqueeze(0)
        self.polar2cart_grid = grid.expand(b, -1, -1, -1) 

        # crate valid pixels mask 
                
        valid_mask = (norm_theta >= -1.0) & (norm_theta <= 1.0) & (norm_r >= -1.0) & (norm_r <= 1.0)
        self.polar2cart_mask = valid_mask.unsqueeze(0).expand(b, -1, -1).float()
        
        
        # --- save first data as init state ---
        # pose as homogenus translation matrix
        self.prev_pose = np.array([[1, 0, init_x], 
                                   [0, 1, init_y], 
                                   [0, 0, 1]])
        
        self.prev_frame = self.polar2car(init_frame)

    @torch.no_grad()
    def forward(self, frame, depth, return_visu = False):

        # --- convert new frame to carthesian ---
        new_frame = self.polar2car(frame)
        
        
        # --- math points with loftr ---
        
        matches = self.match_points({'image0': self.prev_frame, 'mask0': self.polar2cart_mask,
                                     'image1': new_frame, 'mask1': self.polar2cart_mask,
                                    })
        
        pts1 = matches['keypoints0']
        pts2 = matches['keypoints1']
        confidence = matches['confidence']

        # filter with matching confidence 
        valid_matches = confidence > self.pts_match_thresh
        pts1 = pts1[valid_matches]
        pts2 = pts2[valid_matches]

        # --- compensate depth change ---

        # transform to real-world values
        pts1_r = self.scale_px2physcial(pts1)
        pts2_r = self.scale_px2physcial(pts2)

        # # calc distance from sonar to points (acustic ray path)
        # ray1 = torch.sqrt(pts1_r[:, 0]**2 + pts1_r[:, 1]**2)
        # ray2 = torch.sqrt(pts2_r[:, 0]**2 + pts2_r[:, 1]**2)

        # # filtration (discard point swhen acustic ray path is smaller than depth)
        # valid_mask = (ray1 > depth) & (ray2 > depth)
        # pts1_r = pts1_r[valid_mask]
        # pts2_r = pts2_r[valid_mask]
        # ray1 = ray1[valid_mask]
        # ray2 = ray2[valid_mask]
        
        # # calc real distance over ground
        # r1 = torch.sqrt(ray1**2 - depth**2)
        # r2 = torch.sqrt(ray2**2 - depth**2)

        # # extract translation and rotation
        # pts1_r_scaled = pts1_r * (r1 / ray1).unsqueeze(1)
        # pts2_r_scaled = pts2_r * (r2 / ray2).unsqueeze(1)
    
        # --- extract transform matrix - RANSAC ---  
       
        pts1_np = pts1_r.cpu().numpy()
        pts2_np = pts2_r.cpu().numpy()

        M, inlier_mask = cv2.estimateAffinePartial2D(
            pts2_np, pts1_np,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,   
            maxIters=3000,
            confidence=0.999,
        )

        # 
        if M is not None and inlier_mask is not None:
            inlier_mask = inlier_mask.ravel().astype(bool)
            theta = np.arctan2(M[1, 0], M[0, 0])
            tx, ty = M[0, 2], M[1, 2]
            n_in  = inlier_mask.sum()

            local_translation = np.array([[ np.cos(theta), -np.sin(theta), tx],
                                          [ np.sin(theta),  np.cos(theta), ty], 
                                          [ 0,              0,             1]])

            new_pose = self.prev_pose @ local_translation
            
        else:
            inlier_mask = np.zeros(len(pts1_np), dtype=bool)
            new_pose = self.prev_pose

        global_x = new_pose[0, 2]
        global_y = new_pose[1, 2]
        global_azimuth = np.arctan2(new_pose[1, 0], new_pose[0, 0])
        
        if not return_visu: 
            self.prev_frame = new_frame
            self.prev_pose =  new_pose
            return (global_x, global_y), global_azimuth
        
        else: 

            # --- visualisation --- 
            b, c, h, w = new_frame.shape
            frame1_np = self.prev_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame2_np = new_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frames_np = np.concatenate((frame1_np, frame2_np), axis = 1)
            frames_np_rgb = cv2.cvtColor(frames_np, cv2.COLOR_GRAY2RGB)

            visu = {'combined_imgs':frames_np_rgb,
                    'pts1':pts1.detach().cpu().numpy(),
                    'pts2':pts2.detach().cpu().numpy(),
                    'pts2_offset':(0, w)}

            self.prev_frame = new_frame
            self.prev_pose =  new_pose
            return (global_x, global_y), global_azimuth, visu 
        

    @torch.no_grad()
    def polar2car(self, frame, out_shape=None):

        # Sample pixels with grid, padd with zeros
        out_img = F.grid_sample(frame, self.polar2cart_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return out_img

    
    def scale_px2physcial(self, pts_px):
       
        if pts_px.shape[0] == 0:
            return pts_px, torch.zeros(0, dtype=torch.bool, device=self.device)

        u = pts_px[:, 0]
        v = pts_px[:, 1]
        
        out_h, out_w = self.cart_frame_size

        # scale factor 
        scale = (self.r_max - self.r_min) / out_h
         
        # Pixels -> Physicals (maters)
        x = (u - out_w / 2.0) * scale
        y = (out_h - v) * scale + self.r_min
        
        return torch.stack([x, y], dim=1)

    
    
        


    


