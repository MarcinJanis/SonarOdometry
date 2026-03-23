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
from .utils import project_points


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

        # --- init components --- 
        self.PatchGraph = Graph_train(model_config, sonar_config, train_config)
        self.UpdateOperator = Update(model_config)

    def forward(self, x, t, poses_gt, depth_gt, freeze_poses=False):
        
        # --- update graph new data ---
        poses, coords_phi = self.PatchGraph.append(x, t, poses_gt, self.device)

        # --- Optimize iteration --- 
        output_iter = [] # for accumulate output from ech iteration

        for iter in range(self.update_iter):

            # --- detach data optimized in BA from torch graph --- 
            if freeze_poses:
                poses = poses_gt.detach().clone()
            else:
                poses = poses.detach().clone()
                poses.requires_grad_(True)

            coords_phi = coords_phi.detach().clone()
            coords_phi.requires_grad_(True)
            
            # -- get correlation, contexet and graph edges idx -- 
            corr, ctx, source_frame_idx, target_frame_idx, patch_idx, valid_mask = self.PatchGraph.update_step(poses, coords_phi, self.device)

            valid_edges_num = patch_idx.shape[0] # number of active edges

            # --- Optimiziation loop --- 
            if valid_edges_num > 0:

                # --- Get hidden state for active edges --- 
                h = self.PatchGraph.get_hidden_state(valid_mask)

                # --- Update operator --- 
                h, correction = self.UpdateOperator(h, None, corr, ctx, source_frame_idx, target_frame_idx, patch_idx, self.device)
                delta, weights = correction
                
                # --- Bundle adjustement ---
                BA = BundleAdjustment(poses, 
                                      self.PatchGraph.coords_r_theta, 
                                      coords_phi, 
                                      self.sonar_param, 
                                      freeze_poses=freeze_poses)
                
                BA.init_ba(source_frame_idx, target_frame_idx, patch_idx, delta, weights)

                try:
                    opt_poses, opt_elevation = BA.run(max_iter=self.ba_iter, early_stop_tol=self.ba_min_err)
                   
                    # --- Feedback --- 
                    poses = opt_poses
                    coords_phi = opt_elevation 

                    self.PatchGraph.update_hidden_state(h, valid_mask)

                    # --- Final projection error ---
                    if self.training:

                        # Project patches using estimated and real pose and elevation. 
                        b, n, p, _ = self.PatchGraph.coords_r_theta.shape

                        # ground truth patch coords
                        depth_gt_flatten = depth_gt.view(b*n, -1)
                        depth_gt_flatten = depth_gt_flatten[source_frame_idx]

                        r_theta = self.PatchGraph.coords_r_theta.view(b*n*p, -1)
                        r_theta = r_theta[patch_idx, :]
                        depth_r_ratio = torch.clamp(depth_gt_flatten/r_theta[:, 0], -1, 1)
                        gt_elevation = torch.asin(depth_r_ratio) # Approximation!! True only for flat surrounding. 
                        gt_patch_coords = torch.cat((r_theta, gt_elevation), dim=-1)

                        # predicted patch coords
                        opt_elevation_flatten = opt_elevation.view(b*n*p, -1)
                        pred_patch_coords = torch.cat((r_theta, opt_elevation_flatten[patch_idx]), dim=-1)

                        # project with corresponding optimized/gt pose
                        opt_poses_flatten = opt_poses.view(b*n, -1)
                        poses_gt_flatten = poses_gt.view(b*n, -1)
                        predict_patch_coords = project_points(pred_patch_coords, opt_poses_flatten[source_frame_idx], opt_poses_flatten[target_frame_idx])
                        gt_patch_coords = project_points(gt_patch_coords, poses_gt_flatten[source_frame_idx], poses_gt_flatten[target_frame_idx])

                        patch_projection_error = (predict_patch_coords[:, :2] - gt_patch_coords[:, :2])
                        patch_projection_error_pix = self.PatchGraph.scale_phisical2fls(patch_projection_error) # scale to pixels -> normalizaition
                    
                    else:
                        patch_projection_error_pix = None

                except Exception as e: 
                    print(f'[Warning] Cannot run BA for iteration {iter}: {e}')
                    patch_projection_error_pix = torch.Tensor([0, 0]).unsqueeze(0)
                
                output_iter.append((poses, patch_projection_error_pix))
            else: 
                print(f'[Warning] For iteration {iter} there was no active edges. Otimiziation impossible.')

        return  output_iter

            









