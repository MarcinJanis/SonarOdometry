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
from .graph_inference import Graph as Graph_interference
from .graph_training import Graph as Graph_train
from .bundle_adjustment import BundleAdjustment



class DPSO(nn.Module):

    def __init__(self, model_cfg, sonar_cfg, train_cfg, mode = 'inference', output_dir = None):
        super(DPSO, self).__init__()

        self.debug = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- set mode --- 
        if mode == 'train':
            self.train_mode = True
            
        else:
            self.train_mode = False
            

        
        # --- read config files --- 
        with open(model_cfg, "r") as f:
            model_config = Box(yaml.safe_load(f))

        with open(sonar_cfg, "r") as f:
            sonar_config = Box(yaml.safe_load(f))

        with open(train_cfg, "r") as f:
            train_config = Box(yaml.safe_load(f))

        # --- get config parameters --- 
        
        self.update_iter = model_config.UPDATE_ITERATION
        self.ba_iter = model_config.BUNDLE_ADJUSTMENT_ITERATION
        self.ba_min_err = float(model_config.BUNDLE_ADJUSTMENT_MIN_ERR)

            
        # --- init components --- 
        
        if self.train_mode:
            self.train_mode = True
            self.PatchGraph = Graph_train(model_config, sonar_config, train_config)

        else:
            self.train_mode = False
            self.PatchGraph = Graph_interference(model_config, sonar_config, output_dir)

        
        self.UpdateOperator = Update(model_config)
       
        self.hidden_state_dim = model_config.CONTEXT_OUTPUT_CH

    
    
    def debug(self, enable = True):

        if enable:
            self.debug = True
        else:
            self.debug = False
    
    def close(self):
        if not self.train_mode:
            self.PatchGraph.outputf_close()

    def forward(self, x, t):

        # --- Graph --- 
        if self.train_mode:

            # -- update with new data --
            poses, coords_phi = self.PatchGraph.append(x, t, self.device)
            # -- get correlation, contexet and graph edges idx -- 
            corr, ctx, source_frame_idx, target_frame_idx, patch_idx = self.PatchGraph.update_step(poses, coords_phi, self.device)
            # --- init hidden state --- 
            n_edges = patch_idx.shape[0]
            h = torch.zeros((n_edges, self.hidden_state_dim), device=self.device, dtype=torch.float)
            
        else:

            # -- update with new data --
            self.PatchGraph.append(x, t, self.device)
            
            # if self.PatchGraph.frame_n < 6:
            #     return None, None, None 
            
            # -- get correlation, contexet and graph edges idx -- 
            corr, ctx, source_frame_idx, target_frame_idx, patch_idx = self.PatchGraph.update_step(self.device)
            # --- compose hidden state vector for actual edges --- 
            h = self.PatchGraph.get_hidden_state(patch_idx)
        
        # --- Optimize iteration --- 
        for _ in range(self.update_iter):

            # Update operator 
            if patch_idx.shape[0] > 0:
                self.h, correction = self.UpdateOperator(h, None, corr, ctx, source_frame_idx, target_frame_idx, patch_idx, self.device)
                delta, weights = correction
                
                # Bundle adjustement 
                if self.train_mode:
                    BA = BundleAdjustment(poses, self.PatchGraph.coords_r_theta, coords_phi)
                else:
                    BA = BundleAdjustment(self.PatchGraph.poses, self.PatchGraph.patch_coords_r_theta, self.PatchGraph.patch_coords_phi)
                
                BA.init_ba(source_frame_idx, target_frame_idx, patch_idx, delta, weights)
    
                opt_poses, opt_elevation = BA.run(max_iter=self.ba_iter, early_stop_tol=self.ba_min_err)
                
            else:
                if self.train_mode:
                    return None
                else:
                    return None, None, None
        # --- Output ----

        
        if self.train_mode:
            # when training return optimized position for loss fcn
            return  opt_poses
        
        else:
            # save update poses, pts and hidden state to graph 

            self.PatchGraph.update_state(opt_poses.detach().clone(), 
                                         opt_elevation.detach().clone(), 
                                         h.detach().clone(), 
                                         patch_idx)

            # when inference mode return first estimation of current position for control purpose
            pose_vct, time_vct, frame_num = self.PatchGraph.get_state()
            return pose_vct , time_vct, frame_num
            



# def pts_fusion(pts, global_idx):
#     unq_idx, group_idx = torch.unique(global_idx, return_inverse=True)
#     pts_mean = scatter_mean(pts, group_idx, dim=0, dim_size=None)
#     return pts_mean, unq_idx


# def pose_fusion(poses, global_idx):
#     unq_idx, group_idx = torch.unique(global_idx, return_inverse=True)

#     pos_xyz = poses[:, :3]
#     quat_wxyz = poses[:, 3:]

#     mean_pos = scatter_mean(pos_xyz, group_idx, dim=0)
#     mean_quat = scatter_mean(quat_wxyz, group_idx, dim=0)

#     mean_quat = F.normalize(mean_quat, p=2, dim=1)

#     return torch.cat([mean_pos, mean_quat], dim=1), unq_idx
