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
from .graph_training import Graph as Graph_train
from .bundle_adjustment import BundleAdjustment
from .utils import project_points


class DPSO(nn.Module):

    def __init__(self, model_cfg, sonar_cfg, output_dir = None, save_to_file = False):
        super().__init__()
        
        self.debug = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # --- read config files --- 
        with open(model_cfg, "r") as f:
            model_config = Box(yaml.safe_load(f))

        with open(sonar_cfg, "r") as f:
            sonar_config = Box(yaml.safe_load(f))
        self.sonar_param = sonar_config

        # --- get config parameters --- 
        self.update_iter = model_config.UPDATE_ITERATION
        self.ba_iter = model_config.BUNDLE_ADJUSTMENT_ITERATION
        self.hidden_state_dim = model_config.CONTEXT_OUTPUT_CH
        self.ba_min_err = float(model_config.BUNDLE_ADJUSTMENT_MIN_ERR)
        
        # --- init components --- 
        with torch.no_grad():
            self.PatchGraph = Graph_inference(model_config, sonar_config, output_dir, save_to_file)
            
            self.UpdateOperator = Update(model_config)
       
        
    def close(self):
        self.PatchGraph.outputf_close()

    def set_init_pose(self, init_frame, init_t, init_pose):
        self.PatchGraph.set_init_pose(init_pose)
        self.PatchGraph.append(init_frame, init_t, self.device)
        
    def forward(self, x, t):

        self.PatchGraph.append(x, t, self.device)

        # --- Optimize iteration --- 
        output_i = [] # for accumulate output from ech iteration

        for iter in range(self.update_iter):
            
            # -- get correlation, contexet and graph edges idx -- 
            corr, ctx, source_frame_idx, target_frame_idx, patch_idx = self.PatchGraph.update_step(self.device)

            # --- get hidden state for active edges --- 
            h = self.PatchGraph.get_hidden_state(patch_idx)

            # --- Optimiziation loop --- 
            if patch_idx.shape[0] > 0: # if any edge exist

                # --- Update operator --- 
                h, correction = self.UpdateOperator(h, None, corr, ctx, source_frame_idx, target_frame_idx, patch_idx, self.device)
                delta, weights = correction

                # --- Bundle adjustement ---
                BA = BundleAdjustment(self.PatchGraph.actual_poses, 
                                      self.PatchGraph.patch_coords_r_theta, 
                                      self.PatchGraph.patch_coords_phi, 
                                      self.sonar_param,
                                      freeze_poses=False)
                
                BA.init_ba(source_frame_idx, target_frame_idx, patch_idx, delta, weights)
                
                try: 
                    opt_poses, opt_elevation = BA.run(max_iter=self.ba_iter, early_stop_tol=self.ba_min_err)
                    # --- Feedback --- 
                    self.PatchGraph.update_state(opt_poses.detach().clone(), 
                                                    opt_elevation.detach().clone(), 
                                                    h.detach().clone(), 
                                                    patch_idx)
                except Exception as e: 
                    print(f'[Warning] Cannot run BA for frame {self.PatchGraph.frame_n}, iter: {iter}: {e}')

        # --- Output ---
        # return current estimation state for control purpose
        pose_vct, time_vct, frame_num = self.PatchGraph.get_state()
        return pose_vct, time_vct, frame_num
            
