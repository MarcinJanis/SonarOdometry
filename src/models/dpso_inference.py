import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

import numpy as np
import os
import time 

import csv
from box import Box
import yaml

from .update import Update
from .graph_inference import Graph
from .bundle_adjustment import BundleAdjustment
from .logger import DataLogger 
from .utils import project_points, approx_movement, depth_to_elev_angle

class DPSO(nn.Module):

    def __init__(self, model_cfg, sonar_cfg, device, output_data_pth = None):
        super(DPSO, self).__init__()

        self.device = device
            
        # --- read config files --- 
        with open(model_cfg, "r") as f:
            model_config = Box(yaml.safe_load(f))

        with open(sonar_cfg, "r") as f:
            sonar_config = Box(yaml.safe_load(f))

        self.sonar_param = sonar_config

        # --- get config parameters --- 
        self.update_iter = model_config.UPDATE_ITERATION
        self.ba_iter = model_config.BUNDLE_ADJUSTMENT_ITERATION
        # self.ba_min_err = float(model_config.BUNDLE_ADJUSTMENT_MIN_ERR)
        self.motion_appro_model = model_config.MOTION_APPRO_MODEL
        self.patches_per_frame = model_config.PATCHES_PER_FRAME

        self.init_frames = model_config.TIME_WINDOW
        self.freeze_poses_num = model_config.FREEZE_POSES
        self.opticflow_warmup = model_config.OPTICFLOW_WARMUP_ITER

        # --- init components --- 
        self.PatchGraph = Graph(model_config, sonar_config)
        self.UpdateOperator = Update(model_config)

        # --- saving output data inits ---
        
        if not output_data_pth is None:
            self.save_to_file = True
            header_traj = ['pose_no', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw' ]
            heater_pts = ['n', 'x', 'y', 'z']
            self.prim_traj_logger = DataLogger(output_data_pth, 'prim_traj_estim.csv', header_traj, buff_size = 10)
            self.sec_traj_logger = DataLogger(output_data_pth, 'sec_traj_estim.csv', header_traj, buff_size = 10)
            self.pts_logger = DataLogger(output_data_pth, 'pts3d.csv', heater_pts, buff_size = 10)
        else:
            self.save_to_file = False

    def reset(self):
        self.PatchGraph.reset()

    def init_step(self, frame, timestamp, init_pose):

        new_pose = init_pose
        _ = self.PatchGraph.extract_features(frame, new_pose, timestamp)

        
    def forward(self, frame, timestamp, debug_logger=False):
        
        # --- init pose ---
        x_prev, t_prev = self.PatchGraph.get_last_poses(num=2)
        new_pose = approx_movement(x_prev[1], x_prev[0], t_prev[1], t_prev[0], timestamp, 
                                   motion_model=self.motion_appro_model)

        # --- add to graph --- 
        data_poped = self.PatchGraph.extract_features(frame, new_pose, timestamp)
        
        # --- create edges --- 
        self.PatchGraph.create_edges()

        if self.PatchGraph.n > self.init_frames:
            
            # --- optimization loop --- 
            for k in range(self.update_iter): 

                # --- get correlation --- 
                corr, ctx, i_val, j_val, valid_mask = self.PatchGraph.corr(coords_eps=1e-2, device=self.device) 

                # global (edges idx) -> local (buffer idx)

                # patches_idx = i_val
                # src_frames_local_idx = i_val // self.patches_per_frame
                # tgt_frames_local_idx = j_val

                src_frames_local_idx, patches_num = self.PatchGraph.g2l_patch_idx(i_val)
                patches_idx = src_frames_local_idx * self.patches_per_frame + patches_num
                tgt_frames_local_idx = self.PatchGraph.g2l_frame_idx(j_val)
                
                # check if any active edge exist
                val_edges = patches_idx.shape[0]

                if val_edges == 0:
                    print(f'[Warning] There is no active edges. (frame: {self.PatchGraph.n}, updater iteration: {k})')
                    continue

                # --- Update operator --- 
                h = self.PatchGraph.get_hidden_state(valid_mask)
                h, correction = self.UpdateOperator(h, None, corr, ctx, 
                                                    src_frames_local_idx, 
                                                    tgt_frames_local_idx, 
                                                    patches_idx, 
                                                    self.device)
                delta, weights = correction

                self.PatchGraph.update_hidden_state(h, valid_mask)

                # --- Bundle adjustement ---
                if k >= self.opticflow_warmup:
                    poses = self.PatchGraph.get_poses()
                    coords_r_theta, coords_phi = self.PatchGraph.get_patch_coords()
    
                    BA = BundleAdjustment(poses.unsqueeze(0), 
                                        coords_r_theta.unsqueeze(0), 
                                        coords_phi.unsqueeze(0), 
                                        self.sonar_param, 
                                        freeze_poses=self.freeze_poses_num)
                    
                    BA.init_ba(src_frames_local_idx, 
                            tgt_frames_local_idx, 
                            patches_idx, 
                            delta, weights)

                    # try:
                    loss_diff = 0.0
                    opt_poses, opt_phi, loss_diff = BA.run(max_iter=self.ba_iter, 
                                                early_stop_tol=1e-3, 
                                                trust_region=2.0)
                    
                    self.PatchGraph.update_poses(opt_poses.squeeze(0))
                    self.PatchGraph.update_patch_coords(opt_phi.squeeze(0))

                    # except Exception as e: 
                        
                        # print(f'[Warning] Bundle Adjustment failed (frame: {self.PatchGraph.n}, updater iteration: {k}).\n{e}')
                
                else:
                    loss_diff = None

        new_opt_pose, new_timestamp = self.PatchGraph.get_last_poses(num=1)

        if debug_logger: print(f'   - optim iter: {k}, valid edges: {val_edges}, BA loss diff: {loss_diff}')
        
        # --- log data ---
        if self.save_to_file:
            
            prim_traj_data = [self.PatchGraph.n, new_timestamp[0].item()] + new_opt_pose[0].detach().cpu().tolist()
            self.prim_traj_logger.log(prim_traj_data)

            frame_idx, pose_poped, time_poped, patch_idx, patch_coords_poped = data_poped
            
            if frame_idx is not None: 
                
                sec_traj_data = [frame_idx, time_poped.item()] + pose_poped.detach().cpu().tolist()
                self.sec_traj_logger.log(sec_traj_data)

            if patch_idx is not None:

                pts_data = patch_coords_poped.detach().cpu().tolist()
                for i in range(len(patch_idx)):
                    pts_row = [int(patch_idx[i])] + pts_data[i]
                    self.pts_logger.log(pts_row)

        return self.PatchGraph.n, new_timestamp, new_opt_pose

               
