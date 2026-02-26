import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import time 

import csv
from box import Box
import yaml


from .update import Update
from .graph_inference import Graph as Graph_interference
from .graph_training import Graph as Graph_train
from .bundle_adjustment import BundleAdjustment

# file = open('trajectory.csv', mode='w', newline='')
# writer = csv.writer(file)

# # Zapisujemy nagłówki
# writer.writerow(['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
# file.flush() 

# # 2. Główna pętla programu (np. po marginalizacji klatki)
# def log_new_pose(pose_data_list):
#     writer.writerow(pose_data_list)
#     # Wymuszamy fizyczny zapis na dysk bez zamykania pliku!
#     file.flush() 

# # 3. Zakończenie programu (gdy zamykamy system)
# file.close()

class DPSO(nn.Module):

    def _init__(self, model_cfg, sonar_cfg, train_cfg, mode = 'inference', output_dir = None):
        super().__init__()

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


        self.update_iter = model_cfg.UPDATE_ITERATION
        self.ba_iter = model_cfg.BUNDLE_ADJUSTMENT_ITERATION
        self.ba_min_err = model.cfg.BUNDLE_ADJUSTMENT_MIN_ERR
        
        # --- init components --- 
        
        if self.train_mode:
            self.train_mode = True
            self.PatchGraph = Graph_train(model_config, sonar_config, train_config)
        else:
            self.train_mode = False
            self.PatchGraph = Graph_interference(model_config, sonar_config)

        
        self.UpdateOperator = Update(model_config)
       
        self.hidden_state_dim = model_config.CONTEXT_OUTPUT_CH

        # --- create file for output --- 
        if not self.train_mode:
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        primary_traj_file_pth = os.path.join(self.output_dir, "trajectory_primary_estimation.csv")
        secondary_traj_file_pth = os.path.join(self.output_dir, "trajectory_secondary_estimation.csv")
        points3d_file_pth = os.path.join(self.output_dir, "3d_points_estimation.csv")

        # create file writers 
        with open(primary_traj_file_pth, mode = 'w', newline='') as file:
            self.primary_traj_file = file
            self.prim_traj_writer(file)
        with open(secondary_traj_file_pth, mode = 'w', newline='') as file:
            self.secondary_traj_file = file
            self.sec_traj_writer(file)
        with open(points3d_file_pth, mode = 'w', newline='') as file:
            self.points3d_file = file
            self.pts3d__writer(file)

        # init files with headers 
        self.prim_traj_writer.writerow(['pose_no', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw' ])
        self.primary_traj_file.flush()
        self.sec_traj_writer.writerow(['pose_no', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw' ])
        self.secondary_traj_file.flush()
        self.pts3d__writer.writerow(['n', 'x', 'y', 'z'])
        self.points3d_file.flush()
    
    def debug(self, enable = True):

        if enable:
            self.debug = True
        else:
            self.debug = False

    def close_files():
        file.close()

    def forward(self, x, t):

        # --- Graph --- 
        if self.train_mode:
            # -- update with new data --
            poses, coords_phi = self.PatchGraph.append(x, t, self.device)
            # -- get correlation, contexet and graph edges idx -- 
            corr, ctx, source_frame_idx, target_frame_idx, patch_idx = self.PatchGraph.update_step(poses, coords_phi, self.device)
            # --- init hidden state --- 
            n_edges = patch_idx.shape[0]
            self.h = torch.zeros((n_edges, self.hidden_state_dim), device=self.device, dtype=torch.float)
            
        else:
            # -- update with new data --
            self.PatchGraph.append(x, t, self.device)
            # -- get correlation, contexet and graph edges idx -- 
            corr, ctx, source_frame_idx, target_frame_idx, patch_idx = self.PatchGraph.update_step(self.device)
            # --- compose hidden state vector for actual edges --- 
            self.h = self.PatchGraph.get_hidden_state(patch_idx)

        
        # --- Optimize iteration --- 
        for _ in range(self.update_iter):

            # --- Update operator --- 
            self.h, correction = self.UpdateOperator(h, None, corr, ctx, source_frame_idx, target_frame_idx, patch_idx, self.device)
            delta, weights = correction
            
            # --- Bundle adjustement --- 
            if self.train_mode:
                BA = BundleAdjustment(poses, PatchGraph.coords_r_theta, coords_phi)
            else:
                BA = BundleAdjustment(poses, PatchGraph.patch_coords_r_theta, PatchGraph.patch_coords_phi)
            
            BA.init_ba(source_frame_idx, patch_idx, delta, weights)
    
            opt_poses, opt_elevation = BA.run(max_iter=self.ba_iter, early_stop_tol=self.ba_min_err)
            
            # --- Save optimization results --- 

            # pseudo code: 
            # poses[source_frame_idx % buff_size] = opt_poses 
            # patch_state[patch_idx  % buff_size][:, 2] = opt_elevation

        # --- Return current estimation of the newest psition 
        if self.train_mode:
            pass
        else:
            return PatchGraph.get_position()
            
            # add some mechanizm to extract already optimized poses and points clouds,
            # two modes: 
            # 1) when frame is deleting from buffer. We save to file: position and points cloud 
            # 2) return the newest, it will be optimized later but for control purpose it should be enough 

                
            





