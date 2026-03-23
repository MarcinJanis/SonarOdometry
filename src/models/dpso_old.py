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
from .graph_inference import Graph as Graph_inference
from .graph_training_obsolete import Graph as Graph_train
from .bundle_adjustment import BundleAdjustment



class DPSO(nn.Module):

    def __init__(self, model_cfg, sonar_cfg, train_cfg, mode = 'inference', output_dir = None):
        super(DPSO, self).__init__()

        self.debug = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.time_stats = []

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
        self.hidden_state_dim = model_config.CONTEXT_OUTPUT_CH
        self.ba_min_err = float(model_config.BUNDLE_ADJUSTMENT_MIN_ERR)

        # --- init components --- 
        if self.train_mode:
            self.train_mode = True
            self.PatchGraph = Graph_train(model_config, sonar_config, train_config)

        else:
            self.train_mode = False
            self.PatchGraph = Graph_inference(model_config, sonar_config, output_dir)
        
        self.UpdateOperator = Update(model_config)
       
        
    def close(self):
        if not self.train_mode:
            self.PatchGraph.outputf_close()

    def get_time_stats(self):
        time_dict = {
            'graph appending': self.time_stats[1] - self.time_stats[0], 
            'edge correlation': self.time_stats[2] - self.time_stats[1],
            'update operator': self.time_stats[3] - self.time_stats[2],
            'bundle adjustment': self.time_stats[4] - self.time_stats[3],
            'feedback': self.time_stats[5] - self.time_stats[4],
            'total': self.time_stats[5] - self.time_stats[0]
        }
        return time_dict

    def forward(self, x, t, freeze_poses=None, poses_gt=None):

        self.time_stats.append(time.time())
        
        # --- update graph new data ---
        if self.train_mode:
            poses, coords_phi = self.PatchGraph.append(x, t, self.device)
            print(f'Debug 1. poses = {poses.shape}, poses_gt = {poses_gt.shape}')
            if freeze_poses:
                poses = poses_gt
        else:
            self.PatchGraph.append(x, t, self.device)
        self.time_stats.append(time.time())


        # --- Optimize iteration --- 
        output_i = [] # for accumulate output from ech iteration

        for iter in range(self.update_iter):
            
            # -- get correlation, contexet and graph edges idx -- 
            if self.train_mode:
                corr, ctx, source_frame_idx, target_frame_idx, patch_idx = self.PatchGraph.update_step(poses, coords_phi, self.device)
            else:
                corr, ctx, source_frame_idx, target_frame_idx, patch_idx = self.PatchGraph.update_step(self.device)
            self.time_stats.append(time.time())

            # --- get hidden state for active edges --- 
            h = self.PatchGraph.get_hidden_state(patch_idx)

            # --- Optimiziation loop --- 
            if patch_idx.shape[0] > 0: # if any edge exist

                # --- Update operator --- 
                h, correction = self.UpdateOperator(h, None, corr, ctx, source_frame_idx, target_frame_idx, patch_idx, self.device)
                delta, weights = correction
                self.time_stats.append(time.time())

                # --- Bundle adjustement ---
                if self.train_mode:
                    ba_poses = poses
                    ba_r_theta = self.PatchGraph.coords_r_theta
                    ba_phi = coords_phi
                else:
                    ba_poses = self.PatchGraph.actual_poses
                    ba_r_theta = self.PatchGraph.patch_coords_r_theta
                    ba_phi = self.PatchGraph.patch_coords_phi
                   
                BA = BundleAdjustment(ba_poses, ba_r_theta, ba_phi, freeze_poses=freeze_poses)
                BA.init_ba(source_frame_idx, target_frame_idx, patch_idx, delta, weights)
                opt_poses, opt_elevation = BA.run(max_iter=self.ba_iter, early_stop_tol=self.ba_min_err)
                self.time_stats.append(time.time())

                # --- Feedback --- 
                if self.train_mode:
                    poses = opt_poses
                    coords_phi = opt_elevation
                else:
                    self.PatchGraph.update_state(opt_poses.detach().clone(), 
                                                 opt_elevation.detach().clone(), 
                                                 h.detach().clone(), 
                                                 patch_idx)
                    
        # --- Output ----
        if self.train_mode:
            # return optimized position for loss fcn
            self.time_stats.append(time.time())
            return  poses
        else:
            # return current estimation state for control purpose
            pose_vct, time_vct, frame_num = self.PatchGraph.get_state()
            self.time_stats.append(time.time())
            return pose_vct, time_vct, frame_num
            



#TODO:


# - init first position/ scaling position 
# - allow to init buffer with a few frames, and then predict future results 

