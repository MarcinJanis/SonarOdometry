import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

import pypose as pp

import os
import time 

import csv
from box import Box
import yaml

from .update import Update
from .graph_inference import Graph
from .bundle_adjustment import BundleAdjustment

from .utils import approx_movement, transform_to_global, ExtrinsicsCalib

from .logger import DataLogger
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

        self.ba_iter_eval = model_config.BUNDLE_ADJUSTMENT.MAX_ITERATION_EVAL
        self.ba_lr_trans = model_config.BUNDLE_ADJUSTMENT.STEP_TRANSLATION
        self.ba_lr_rot = model_config.BUNDLE_ADJUSTMENT.STEP_ROTATION
        self.ba_lr_elev = model_config.BUNDLE_ADJUSTMENT.STEP_ELEV
        self.ba_patience = model_config.BUNDLE_ADJUSTMENT.PATIENCE

        self.freeze_poses_num = model_config.BUNDLE_ADJUSTMENT.FREEZE_POSES

        self.motion_appro_model = model_config.MOTION_APPRO_MODEL
        self.patches_per_frame = model_config.PATCHES_PER_FRAME

        self.init_frames = model_config.TIME_WINDOW

        self.buff_size = model_config.BUFF_SIZE

        # --- init components --- 
        self.PatchGraph = Graph(model_config, sonar_config)
        self.UpdateOperator = Update(model_config)

        self.calib = ExtrinsicsCalib(T = [sonar_config.position.x, sonar_config.position.y, sonar_config.position.z],
                                     R = [sonar_config.position.roll, sonar_config.position.pitch, sonar_config.position.yaw])


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

    def close(self):
        if self.save_to_file:
            self.prim_traj_logger.close()
            self.sec_traj_logger.close()
            self.pts_logger.close()
            
    def init_step(self, frame, timestamp, init_pose):

        new_pose = init_pose
        _ = self.PatchGraph.extract_features(frame, new_pose, timestamp) 
        self.PatchGraph.create_edges()

        prim_traj_data = [self.PatchGraph.n, timestamp.item()] + init_pose.detach().cpu().tolist()
        self.prim_traj_logger.log(prim_traj_data)
        
    @torch.no_grad()
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
                
                # --- get correlation --- 
                corr, ctx, patches_global_idx, tgt_frames_global_idx, valid_mask = self.PatchGraph.corr(coords_eps=1e-2, device=self.device) 

                src_frames_local_idx, patches_local_idx = self.PatchGraph.g2l_patch_idx(patches_global_idx)
                src_frames_global_idx = patches_global_idx // self.patches_per_frame
                tgt_frames_local_idx = self.PatchGraph.g2l_frame_idx(tgt_frames_global_idx)

                # check if any active edge exist
                # val_edges = patches_global_idx.shape[0]

                # --- Update operator --- 
                h = self.PatchGraph.get_hidden_state()
                h, correction = self.UpdateOperator(h, None, corr, ctx, src_frames_global_idx, tgt_frames_global_idx, patches_global_idx, self.device)
                
                delta, weights = correction
                weights = weights * valid_mask.view(-1, 1)
                
                # weights = torch.ones_like(weights) * valid_mask.view(-1, 1) # tmp

                self.PatchGraph.update_hidden_state(h)

                # --- Bundle adjustement ---

                # get chronological graph state
                poses, coords_r_theta, coords_phi, shifts = self.PatchGraph.get_graphstate(chronological = True)
                
                # shift local buff indexes
                src_frames_local_idx_chrono = (src_frames_local_idx + shifts) % self.buff_size
                tgt_frames_local_idx_chrono = (tgt_frames_local_idx + shifts) % self.buff_size
                patches_chrono_idx = src_frames_local_idx_chrono * self.patches_per_frame + patches_local_idx
                
                BA = BundleAdjustment(poses.unsqueeze(0),
                                    coords_r_theta.unsqueeze(0), coords_phi.unsqueeze(0), 
                                    src_frames_local_idx_chrono, tgt_frames_local_idx_chrono, patches_chrono_idx,
                                    delta, weights,
                                    self.sonar_param, self.freeze_poses_num)
                BA.to(self.device)


                
                poses_optimized, elevation_optimized = BA.run(max_iter = self.ba_iter_eval, 
                                                            patience = self.ba_patience, 
                                                            min_delta = 1e-4,
                                                            lr_elev=self.ba_lr_elev, lr_rot=self.ba_lr_rot, lr_trans = self.ba_lr_trans,
                                                            disp_stats=False)
    

                self.PatchGraph.update_graphstate(poses_optimized.squeeze(0), elevation_optimized.squeeze(0), shifts = -shifts)
                
                # self.PatchGraph.update_poses(poses_optimized.squeeze(0))
                # self.PatchGraph.update_patch_coords(elevation_optimized.squeeze(0))

               
        # get latest optimized pose 
        new_opt_pose, new_timestamp = self.PatchGraph.get_last_poses(num=1)

        # --- log data ---
        if self.save_to_file:
            
            prim_traj_data = [self.PatchGraph.n, new_timestamp[0].item()] + new_opt_pose[0].detach().cpu().tolist()
            self.prim_traj_logger.log(prim_traj_data)

            frame_idx, pose_poped, time_poped, patch_idx, patch_coords_poped = data_poped
        
            # transform poped points from local frame to global frame
            if frame_idx is not None: 
        
                sec_traj_data = [frame_idx, time_poped.item()] + pose_poped.detach().cpu().tolist()
                self.sec_traj_logger.log(sec_traj_data)

            if patch_idx is not None:
                patch_coords_glob_poped = transform_to_global(patch_coords_poped, 
                                                              pose_poped.unsqueeze(0).expand(patch_coords_poped.shape[0], 7))
                
                pts_data = patch_coords_glob_poped.detach().cpu().tolist()
                for i in range(len(patch_idx)):
                    pts_row = [int(patch_idx[i])] + pts_data[i]
                    self.pts_logger.log(pts_row)

        return self.PatchGraph.n, new_timestamp, new_opt_pose

               
