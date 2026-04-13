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

from .utils import project_points, approx_movement, depth_to_elev_angle, ExtrinsicsCalib

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

        # self.update_iter = model_config.UPDATE_ITERATION
        self.ba_iter = model_config.BUNDLE_ADJUSTMENT.MAX_ITERATION
        self.ba_lr_trans = model_config.BUNDLE_ADJUSTMENT.STEP_TRANSLATION
        self.ba_lr_rot = model_config.BUNDLE_ADJUSTMENT.STEP_ROTATION
        self.ba_lr_elev = model_config.BUNDLE_ADJUSTMENT.STEP_ELEV
        self.ba_patience = model_config.BUNDLE_ADJUSTMENT.PATIENCE

        self.freeze_poses_num = model_config.FREEZE_POSES.FREEZE_POSES
        
        self.motion_appro_model = model_config.MOTION_APPRO_MODEL
        self.patches_per_frame = model_config.PATCHES_PER_FRAME
        
        self.init_frames = init_frames
        self.batch_size = batch_size
        self.frames_in_series = frames_in_series

        assert frames_in_series >= init_frames, f'[Error] Frames number for initialization shall be equal or smaller than total frames number'

        # --- init components --- 
        self.PatchGraph = Graph(model_config, sonar_config, batch_size, frames_in_series)
        self.UpdateOperator = Update(model_config)
        self.calib = ExtrinsicsCalib(T = [sonar_config.position.x, sonar_config.position.y, sonar_config.position.z],
                                     R = [sonar_config.position.roll, sonar_config.position.pitch, sonar_config.position.yaw])

    def forward(self, frames, timestamp, poses_gt, depth_gt, supervised, freeze_poses=False, init_poses_noise = 0.0, debug_logger=False):
        
        device = frames.device
        
        self.PatchGraph.reset()

        batch_size, frames_max = frames.shape[:2]

        # --- Extrinsic calibration --- 
        # gt -> sonar frame
        poses_gt = self.calib.pose(poses_gt)
        depth_gt = self.calb.depth(depth_gt)
        
        # --- Graph Init --- 

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

        # init edges
        self.PatchGraph.init_edges(self.init_frames, device)

        # --- Iterate over sequence --- 
        output_iter = [] 
        
        for i in range(frames_max): 
            if debug_logger: print(f'=== Processing: frame {i+1}/{frames_max} ===')
            
            # time measurement
            torch.cuda.synchronize()
            t0 = time.time()

            # --- Graph Append ---
            # if init is done, append graph with new edges
            if i >= self.init_frames: 

                x1, x2 = poses[:, i-2, :], poses[:, i-1, :]
                t1, t2, t3 = timestamp[:, i-2], timestamp[:, i-1], timestamp[:, i]
                new_pose = approx_movement(x1, x2, t1, t2, t3, 
                                           motion_model=self.motion_appro_model)
                 
                poses = torch.cat([poses, new_pose.unsqueeze(1)], dim=1)

                self.PatchGraph.create_new_edges(i, device)

            # detach from torch graph for each frame in sequence
            poses = poses.detach()
            coords_phi = coords_phi.detach()

            torch.cuda.synchronize()
            t1 = time.time()

            # --- Optimization --- 
            
            # --- Correlation --- 
            corr, ctx, patches_idx, tgt_frames_idx, valid_mask = self.PatchGraph.corr(poses, coords_phi, coords_eps=1e-2, device=device)
            src_frames_idx = patches_idx // self.patches_per_frame

            # force zero correlation for non valid edges 
            corr = corr * valid_mask.view(-1, 1)

            # check if any active edge exist
            val_edges = torch.sum(valid_mask)
            # if debug_logger: print(f'   - optim iter: {k}, valid edges: {val_edges}')
            
            torch.cuda.synchronize()
            t2 = time.time()

            # --- Update operator --- 
            h = self.PatchGraph.get_hidden_state()
            h, correction = self.UpdateOperator(h, None, corr, ctx, src_frames_idx, tgt_frames_idx, patches_idx, device)
            
            delta, weights = correction

            self.PatchGraph.update_hidden_state(h)
            
            torch.cuda.synchronize()
            t3 = time.time()

            # --- Bundle Adjustement ---
            if freeze_poses:
                b, n, _ = poses.shape
                ba_freeze_poses = b*n
            else:
                ba_freeze_poses = self.freeze_poses_num

            # detach all tensors passed to BA
            BA = BundleAdjustment(supervised, poses.detach(),
                                coords_r_theta.detach(), coords_phi.detach(), 
                                src_frames_idx.detach(), tgt_frames_idx.detach(), patches_idx.detach(),
                                delta.detach(), weights.detach(),
                                self.sonar_param, ba_freeze_poses)
            BA.to(device)

            # try:
            with torch.no_grad():
                poses_optimized, elevation_optimized = BA.run(max_iter= self.ba_iter, 
                                                              patience = self.ba_patience, 
                                                              min_delta = 1e-4,
                                                              lr_elev=self.ba_lr_elev, lr_rot=self.ba_lr_rot, lr_trans = self.ba_lr_trans,
                                                              disp_stats=False)

            # feedback after BA
            poses = poses_optimized
            coords_phi = elevation_optimized

            torch.cuda.synchronize()
            t4 = time.time()

            # --- Reprojection error ---
            physic2fls_scale_factor = torch.tensor([self.sonar_param.resolution.bins / (self.sonar_param.range.max - self.sonar_param.range.min),
                                                    self.sonar_param.resolution.beams / self.sonar_param.fov.horizontal], device = device).view(1, 2)
            
            b, n, p, _ = coords_r_theta.shape
            coords_r_theta_expand = coords_r_theta.view(b*n*p, 2)[patches_idx]

            if supervised:
                ref_poses = poses_gt
                depth_gt_expand = depth_gt.view(b*n)[src_frames_idx]
                r_expand = coords_r_theta_expand[:, 0]
                ref_phi =  depth_to_elev_angle(depth_gt_expand, r_expand)
                ref_phi = ref_phi.unsqueeze(1)
            else:
                ref_poses = poses_optimized
                ref_phi = elevation_optimized.view(b*n*p, 1)[patches_idx]
                
            
            origin_points = torch.cat([coords_r_theta_expand, ref_phi], dim=1).detach() # detach(), bec its reference val to loss!

            b, n_act, _ = ref_poses.shape
            origin_poses = ref_poses.view(b*n_act, 7)[src_frames_idx, :]
            target_poses = ref_poses.view(b*n_act, 7)[tgt_frames_idx, :]

            ref_projection = project_points(origin_points, origin_poses, target_poses)        
            
            ref_projection = ref_projection[:, :2] * physic2fls_scale_factor
            pred_projection = coords_r_theta_expand * physic2fls_scale_factor + delta 

            output_iter.append((poses, ref_projection, pred_projection, valid_mask))
            
            torch.cuda.synchronize()
            t5 = time.time()
            
            # Time measurement results
            print(f"Frame {i:02d} | Graph: {t1-t0:.4f}s | Corr: {t2-t1:.4f}s | Update: {t3-t2:.4f}s | BA: {t4-t3:.4f}s | Reproj: {t5-t4:.4f}s | TOTAL: {t5-t0:.4f}s")
        
        return output_iter
