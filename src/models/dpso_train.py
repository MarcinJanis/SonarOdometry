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

        self.init_frames = init_frames
        self.batch_size = batch_size
        self.frames_in_series = frames_in_series

        assert frames_in_series >= init_frames, f'[Error] Frames number for initialization shall be equal or smaller than total frames number'

        # --- init components --- 
        self.PatchGraph = Graph(model_config, sonar_config, batch_size, frames_in_series)
        self.UpdateOperator = Update(model_config)


    def forward(self, frames, timestamp, poses_gt, depth_gt, freeze_poses=False, init_poses_noise = 0.0, debug_logger=False):
        
        self.PatchGraph.reset()

        batch_size, frames_max = frames.shape[:2]

        # --- graph init --- 

        # global features extractor on whole sequence
        coords_phi, coords_r_theta  = self.PatchGraph.extract_features(frames, self.device) 
        
        # init poses and time  stamps
        if freeze_poses:
            poses = poses_gt[:, :self.init_frames, :]
        else: 
            poses = poses_gt[:, :self.init_frames, :]
            noise_translation = torch.rand_like(poses[:, :, :3]) * init_poses_noise
            noise_rotation = torch.rand_like(poses[:, :, 3:]) * 0.1
            noise = torch.cat([noise_translation, noise_rotation], dim=-1)
            poses = poses + noise
            poses[:, :, 3:] = F.normalize(poses[:, :, 3:], p=2, dim=-1)

        time = timestamp

        # init edges
        self.PatchGraph.init_edges(self.init_frames, self.device)

        # iteration over sequence
        output_iter = [] 
        
        for i in range(frames_max): # iterate over all frames

            if i >= self.init_frames: # if init is done, append graph with new edges

                x1, x2 = poses[:, i-2, :], poses[:, i-1, :]
                t1, t2, t3 = time[:, i-2], time[:, i-1], time[:, i]
                new_pose = approx_movement(x1, x2, t1, t2, t3, 
                                           motion_model=self.motion_appro_model)
                 
                if debug_logger: print(f'  New pose added: {new_pose}')
                poses = torch.cat([poses, new_pose.unsqueeze(1)], dim=1)

                self.PatchGraph.create_new_edges(i, self.device)

            # detach potimized parameters from graph for BA 
            poses = poses.detach()
            poses.requires_grad_(True)
            coords_phi = coords_phi.detach()
            coords_phi.requires_grad_(True)

            # --- optimization loop --- 
            for k in range(self.update_iter): 
                if debug_logger: print(f'Processing: frame {i}/{frames_max}')
                # --- get correlation --- 
            
                corr, ctx, i_val, j_val, valid_mask = self.PatchGraph.corr(poses, coords_phi, coords_eps=1e-2, device=self.device) # tu chyba też trzeba bedzie podać i !!!!

                patches_idx = i_val
                src_frames_idx = i_val // self.patches_per_frame
                tgt_frames_idx = j_val

                # check if any active edge exist
                val_edges = patches_idx.shape[0]
                if debug_logger: print(f'   > valid edges: {val_edges}')

                if val_edges == 0:

                    output_iter.append((poses, None, None))

                    print(f'[Warning] There is no active edges. (frame: {i}, updater iteration: {k})')
                    continue

                # --- Update operator --- 
                h = self.PatchGraph.get_hidden_state(valid_mask)
                h, correction = self.UpdateOperator(h, None, corr, ctx, src_frames_idx, tgt_frames_idx, patches_idx, self.device)
                delta, weights = correction

                self.PatchGraph.update_hidden_state(h, valid_mask)

                # --- Bundle adjustement ---
                BA = BundleAdjustment(poses, 
                                      coords_r_theta, 
                                      coords_phi, 
                                      self.sonar_param, 
                                      freeze_poses=freeze_poses)
                
                BA.init_ba(src_frames_idx, tgt_frames_idx, patches_idx, delta, weights)

                try:
                    opt_poses, opt_phi = BA.run(max_iter=self.ba_iter, 
                                                early_stop_tol=1e-3, 
                                                trust_region=2.0)
                    poses = opt_poses
                    coords_phi = opt_phi

                except Exception as e: 
                    
                    print(f'[Warning] Bundle Adjustment failed (frame: {i}, updater iteration: {k}).\n{e}')
            
            # --- calc patch coords with gt poses and predicted poses --- 
            b, n, p, _ = coords_r_theta.shape
        
            coords_r_theta_flat = coords_r_theta.view(b*n*p, -1)
            coords_r_theta_expand = coords_r_theta_flat[patches_idx]

            opt_phi_expand = coords_phi.view(b*n*p)[patches_idx]
            depth_gt_expand = depth_gt.view(b*n)[src_frames_idx]
         
            gt_phi = depth_to_elev_angle(depth_gt_expand, coords_r_theta_expand[:, 0])

            pred_patch_coords = torch.cat([coords_r_theta_expand, opt_phi_expand.unsqueeze(-1)], dim=1)
            gt_patch_coords = torch.cat([coords_r_theta_expand, gt_phi.unsqueeze(-1)], dim=1)

            # opt and gt poses only up to current frame
            b, n_act, _ = poses.shape
            opt_pose_flat = poses.view(b*n_act, -1)
            gt_pose_flat = poses_gt[:, :n_act, :].view(b*n_act, -1)

            projected_coords_pred = project_points(pred_patch_coords, 
                                                   opt_pose_flat[src_frames_idx], 
                                                   opt_pose_flat[tgt_frames_idx])
            
            projected_coords_gt = project_points(gt_patch_coords, 
                                                 gt_pose_flat[src_frames_idx], 
                                                 gt_pose_flat[tgt_frames_idx])

            pred_fls_coords  = self.PatchGraph.scale_phisical2fls(projected_coords_pred)
            gt_fls_coords = self.PatchGraph.scale_phisical2fls(projected_coords_gt)

            output_iter.append((poses, gt_fls_coords, pred_fls_coords))
        
        return output_iter

               
