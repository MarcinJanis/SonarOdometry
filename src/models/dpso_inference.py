import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

import os
import time 

import csv
from box import Box
import yaml

from .update import Update
from .graph_inference import Graph
from .bundle_adjustment import BundleAdjustment

from .utils import project_points, approx_movement, depth_to_elev_angle

class DPSO_train(nn.Module):

    def __init__(self, model_cfg, sonar_cfg, batch_size, frames_in_series, init_frames, device):
        super(DPSO_train, self).__init__()

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

        # --- init components --- 
        self.PatchGraph = Graph(model_config, sonar_config, batch_size, frames_in_series)
        self.UpdateOperator = Update(model_config)

    def reset(self):
        self.PatchGraph.reset()

    def save_to_file(self, data):
        pass

    def init_step(self, frame, timestamp, init_pose):

        new_pose = init_pose
        _ = self.PatchGraph.extract_features(frame, new_pose, timestamp)

        
    def forward(self, frame, timestamp, debug_logger=False):
        
        # --- init pose ---
        x_prev, t_prev = self.PatchGraph.get_last_poses(n=2)
        new_pose = approx_movement(x_prev[1], x_prev[0], t_prev[1], t_prev[0], timestamp, 
                                   motion_model=self.motion_appro_model)

        # --- add to graph --- 
        data_poped = self.PatchGraph.extract_features(frame, new_pose, timestamp)
        self.save_to_file(data_poped)
        
        # --- create edges --- 
        self.PatchGraph.create_edges()

        if self.PatchGraph.n > self.init_frames:
            
            # --- optimization loop --- 
            for k in range(self.update_iter): 

                # --- get correlation --- 
                corr, ctx, i_val, j_val, valid_mask = self.PatchGraph.corr(coords_eps=1e-2, device=self.device) 

                patches_idx = i_val
                src_frames_idx = i_val // self.patches_per_frame
                tgt_frames_idx = j_val

                # check if any active edge exist
                val_edges = patches_idx.shape[0]

                if val_edges == 0:
                    print(f'[Warning] There is no active edges. (frame: {i}, updater iteration: {k})')
                    continue

                # --- Update operator --- 
                h = self.PatchGraph.get_hidden_state(valid_mask)
                h, correction = self.UpdateOperator(h, None, corr, ctx, src_frames_idx, tgt_frames_idx, patches_idx, self.device)
                delta, weights = correction

                self.PatchGraph.update_hidden_state(h, valid_mask)

                # --- Bundle adjustement ---

                poses = self.PatchGraph.get_poses()
                coords_r_theta, coords_phi = self.PatchGraph.get_patch_coords()
 
                BA = BundleAdjustment(poses, 
                                      coords_r_theta, 
                                      coords_phi, 
                                      self.sonar_param, 
                                      freeze_poses=False)
                
                BA.init_ba(src_frames_idx, tgt_frames_idx, patches_idx, delta, weights)

                try:
                    loss_diff = 0.0
                    opt_poses, opt_phi, loss_diff = BA.run(max_iter=self.ba_iter, 
                                                early_stop_tol=1e-3, 
                                                trust_region=2.0)
                    

                    self.PatchGraph.update_poses(opt_poses)
                    b, n, p, _ = opt_phi.shape
                    self.PatchGraph.update_patch_coords(opt_phi.view(b*n, p, 1))

                except Exception as e: 
                    
                    print(f'[Warning] Bundle Adjustment failed (frame: {self.PatchGraph.n}, updater iteration: {k}).\n{e}')
            
                if debug_logger: print(f'   - optim iter: {k}, valid edges: {val_edges}, BA loss diff: {loss_diff}')
        
          
        new_opt_pose = self.PatchGraph.get_last_poses(n=1)

        return new_opt_pose

               
