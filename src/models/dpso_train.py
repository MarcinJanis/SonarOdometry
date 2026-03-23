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
from .graph_train import Graph
from .bundle_adjustment import BundleAdjustment

from .utils import project_points, approx_movement

class DPSO_train(nn.Module):

    def __init__(self, model_cfg, sonar_cfg, train_cfg):
        super(DPSO_train, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # --- read config files --- 
        with open(model_cfg, "r") as f:
            model_config = Box(yaml.safe_load(f))

        with open(sonar_cfg, "r") as f:
            sonar_config = Box(yaml.safe_load(f))
        self.sonar_param = sonar_config

        with open(train_cfg, "r") as f:
            train_config = Box(yaml.safe_load(f))


        # --- get config parameters --- 
        self.update_iter = model_config.UPDATE_ITERATION
        self.ba_iter = model_config.BUNDLE_ADJUSTMENT_ITERATION
        self.hidden_state_dim = model_config.CONTEXT_OUTPUT_CH
        self.ba_min_err = float(model_config.BUNDLE_ADJUSTMENT_MIN_ERR)

        self.init_frames = model_config.INIT_FRAMES

        # --- init components --- 
        self.PatchGraph = Graph(model_config, sonar_config, train_config)
        self.UpdateOperator = Update(model_config)


    def forward(self, frames, timestamp, poses_gt, depth_gt, freeze_poses=False):

        b, frames_max, ... = frames.shape

        # --- graph init --- 

        # global features extractor on whole sequence
        elev_angle = self.PatchGraph.extract_features(frames, self.device) 

        # init poses
        poses = poses_gt[:, :self.init_frames, :]

        # init edges
        self.PatchGraph.init_edges()

        # iteration over sequence
        output_iter = [] 

        for i in range(frames_max): # iterate over all frames

            if i >= self.init_frames: # if init is done, append graph with new edges
                approx_movement() # nie wiem co z tym !!!!
                self.PatchGraph.create_new_edges(i, self.device)

            # --- optimization loop --- 
            for k in range(self.update_iter): 

                # --- get correlation --- 
                corr, ctx, i_val, j_val, valid_mask = self.PatchGraph.corr(poses, elev_angle, eps=1e-2) # tu chyba też trzeba bedzie podać i !!!!

                patches_idx = i_val
                src_frames_idx = i_val // self.patches_per_frame
                tgt_frames_idx = j_val

                # --- Update operator --- 
                h, correction = self.UpdateOperator(h, None, corr, ctx, src_frames_idx, tgt_frames_idx, patches_idx, self.device)
                delta, weights = correction

                # --- Bundle adjustement ---
                BA = BundleAdjustment(poses, 
                                      self.PatchGraph.coords_r_theta, 
                                      elev_angle, 
                                      self.sonar_param, 
                                      freeze_poses=freeze_poses)
                
                BA.init_ba(src_frames_idx, tgt_frames_idx, patches_idx, delta, weights)

                try:
                    opt_poses, opt_elevation = BA.run(max_iter=self.ba_iter, early_stop_tol=self.ba_min_err)
                
                    # feed back with poses elevation angle and hidden state 

                except Exception as e: 
                    pass
                    # exception what to do with BA 

                # or calc projection error or better - return necessery thing to calc it 
