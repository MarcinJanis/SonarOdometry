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

    def __init__(self, model_cfg, sonar_cfg, batch_size, frames_in_series, init_frames):
        super(DPSO_train, self).__init__()

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
        self.freeze_poses_num = model_config.FREEZE_POSES

        self.init_frames = init_frames
        self.batch_size = batch_size
        self.frames_in_series = frames_in_series

        assert frames_in_series >= init_frames, f'[Error] Frames number for initialization shall be equal or smaller than total frames number'

        # --- init components --- 
        self.PatchGraph = Graph(model_config, sonar_config, batch_size, frames_in_series)
        self.UpdateOperator = Update(model_config)


    def forward(self, frames, timestamp, poses_gt, depth_gt, supervised, freeze_poses=False, init_poses_noise = 0.0, debug_logger=False):
        
        device = frames.device
        
        self.PatchGraph.reset()

        batch_size, frames_max = frames.shape[:2]

        # --- graph init --- 

        # global features extractor on whole sequence
        coords_phi, coords_r_theta  = self.PatchGraph.extract_features(frames, device) 
        
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
        self.PatchGraph.init_edges(self.init_frames, device)

        # iteration over sequence
        output_iter = [] 
        
        for i in range(frames_max): # iterate over all frames

            if debug_logger: print(f'Processing: frame {i}/{frames_max-1}')

            if i >= self.init_frames: # if init is done, append graph with new edges

                x1, x2 = poses[:, i-2, :], poses[:, i-1, :]
                t1, t2, t3 = time[:, i-2], time[:, i-1], time[:, i]
                new_pose = approx_movement(x1, x2, t1, t2, t3, 
                                           motion_model=self.motion_appro_model)
                 
                poses = torch.cat([poses, new_pose.unsqueeze(1)], dim=1)

                self.PatchGraph.create_new_edges(i, device)

            # detach hidden state from graph - reset for new frame
            if not self.PatchGraph.hidden_state is None:
                self.PatchGraph.hidden_state = self.PatchGraph.hidden_state.detach()


            # --- optimization loop --- 
            for k in range(self.update_iter): 

                # detach from graph poses and coords_phi 
                poses = poses.detach()
                coords_phi = coords_phi.detach()

                # --- get correlation --- 
                corr, ctx, patches_idx, tgt_frames_idx, valid_mask = self.PatchGraph.corr(poses, coords_phi, coords_eps=1e-2, device=device)
                src_frames_idx = patches_idx // self.patches_per_frame

                # force zero correlation for non valid edges 
                corr = corr * valid_mask.view(-1, 1)

                # check if any active edge exist
                val_edges = torch.sum(valid_mask)
                if  val_edges == 0 and debug_logger: 
                    print(f'[Warning] There is no active edges. (frame: {i}, updater iteration: {k})')
                    # continue

                # --- Update operator --- 
                h = self.PatchGraph.get_hidden_state()
                h, correction = self.UpdateOperator(h, None, corr, ctx, src_frames_idx, tgt_frames_idx, patches_idx, device)
                delta, weights = correction

                self.PatchGraph.update_hidden_state(h)

                # --- Bundle adjustement ---
                if freeze_poses:
                    b, n, _ = poses.shape
                    ba_freeze_poses = b*n
                else:
                    ba_freeze_poses = self.freeze_poses_num

                BA = BundleAdjustment(poses, supervised,
                                    poses_gt, depth_gt,
                                    coords_r_theta, coords_phi, 
                                    src_frames_idx, tgt_frames_idx, patches_idx,
                                    delta, weights,
                                    self.sonar_param, ba_freeze_poses)
                BA.to(device)

                try:
                    pose_optimized, elevation_optimized, predicted_projection, target_projection = BA.run(max_iter=self.ba_iter, 
                                                                                                    early_stop_tol=1e-3, 
                                                                                                    trust_region=2.0)
                    poses = pose_optimized
                    coords_phi = elevation_optimized

                except Exception as e:     
                    print(f'[Warning] Bundle Adjustment failed (frame: {i}, updater iteration: {k}).\n{e}')
            
                if debug_logger: print(f'   - optim iter: {k}, valid edges: {val_edges}')
        
            output_iter.append((poses, target_projection, predicted_projection, valid_mask))
        
        return output_iter





