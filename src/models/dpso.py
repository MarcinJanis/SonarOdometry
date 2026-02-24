import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import time 

from box import Box
import yaml


from .update import Update
from .graph_inference import Graph as Graph_interference
from .graph_training import Graph as Graph_train
from .bundle_adjustment import BundleAdjustment



class DPSO(nn.Module):

    def _init__(self, model_cfg, sonar_cfg, train_cfg, mode = 'inference'):
        super().__init__()

        self.debug = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- read config files --- 
        with open(model_cfg, "r") as f:
            model_config = Box(yaml.safe_load(f))

        with open(sonar_cfg, "r") as f:
            sonar_config = Box(yaml.safe_load(f))

        with open(train_cfg, "r") as f:
            train_config = Box(yaml.safe_load(f))


        self.update_iter = model_cfg.UPDATE_ITERATION

        # --- init components --- 
        if mode == 'inference':
            self.train_mode = False
            self.PatchGraph = Graph_interference(model_config, sonar_config)
            
        elif mode == 'train':
            self.train_mode = True
            self.PatchGraph = Graph_train(model_config, sonar_config, train_config)

        else:
            raise 'Wrong mode. Choose \'inference\' or \'train\'.'
        
        self.UpdateOperator = Update(model_config)
       
        self.hidden_state_dim = model_config.CONTEXT_OUTPUT_CH
        
    def debug(self, enable = True):

        if enable:
            self.debug = True
        else:
            self.debug = False
    

    def forward(self, x, t):

        # --- Graph --- 
        if self.train_mode:
            # -- update with new data --
            poses, coords_phi = self.PatchGraph.append(x, t, self.device)
            # -- get correlation, contexet and graph edges idx -- 
            corr, ctx, source_frame_idx, target_frame_idx, patch_idx = self.PatchGraph.update_step(poses, coords_phi, self.device)
        else:
            # -- update with new data --
            self.PatchGraph.append(x, t, self.device)
            # -- get correlation, contexet and graph edges idx -- 
            corr, ctx, source_frame_idx, target_frame_idx, patch_idx = self.PatchGraph.update_step(self.device)
        
        


        # --- Optimize iteration --- 

        # -- init hidden state -- 
        self.h = torch.zeros((n_edges, self.hidden_state_dim), device=self.device, dtype=torch.float) 
        # -- iterate -- 
        for _ in range(self.update_iter):

            # --- Update operator --- 
            self.UpdateOperator

            # --- Bundle adjustement --- 
            
            # n_edges = corr_t.shape[0]
            # h_init = torch.zeros((n_edges, model_config.CONTEXT_OUTPUT_CH), device=device, dtype=torch.float)

            h, correction = self.UpdateOperator(h, None, corr, ctx, source_frame_idx, target_frame_idx, patch_idx, self.device)
            delta, weights = correction 