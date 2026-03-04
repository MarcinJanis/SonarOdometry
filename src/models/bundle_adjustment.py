import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import transorm_points_coords, projection_type, transform_matrix

import pypose as pp


class BundleAdjustment(nn.Module):
    def __init__(self, poses, patch_coords_r_theta, patch_coords_phi):
        super().__init__()
        
        # self.buff_size = poses.shape[1] # get buff size

        # --- set propper shape --- 
        if len(patch_coords_r_theta.shape) == 4:
            
            b, n, p, _ = patch_coords_r_theta.shape
            self.b = b
            self.n = n
            self.p = p 

        elif len(patch_coords_r_theta.shape) == 3:
            
            bn, p, _ = patch_coords_r_theta.shape
            self.b = 1
            self.n = bn
            self.p = p 

        self.pose_num = self.b*self.n
        self.edge_num = self.b*self.n*self.p

        poses = poses.view(1, self.pose_num, 7)
        patch_coords_r_theta = patch_coords_r_theta.view(1, self.edge_num, 2)
        patch_coords_phi = patch_coords_phi.view(1, self.edge_num, 1)

        # --- define parameters to optimize ---
        poses_se3 = pp.SE3(poses)
        self.poses = pp.Parameter(poses_se3)

        self.elevation_angle = nn.Parameter(patch_coords_phi) # pp.Parameter(patch_coords_phi)

        # --- define constants parameters --- 
        self.patch_coords = patch_coords_r_theta
        


    def transform(self, source_poses, target_poses, coords):

        source_poses = source_poses.squeeze(0)
        target_poses = target_poses.squeeze(0)
        coords = coords.squeeze(0)

        local_source_coords = transorm_points_coords(coords, projection_type.POLAR2CARTESIAN)

        global_coords = source_poses @ local_source_coords

        local_target_coords = target_poses.Inv() @ global_coords

        coords = transorm_points_coords(local_target_coords, projection_type.CARTESIAN2POLAR)

        # source_poses = source_poses.unsqueeze(0)
        # target_poses = target_poses.unsqueeze(0)
        # coords = coords

        return coords.unsqueeze(0)
    


    def init_ba(self, source_poses_idx, target_poses_idx, patch_idx, delta, weights):
        
        self.source_poses_idx = source_poses_idx % self.pose_num
        self.target_poses_idx = target_poses_idx % self.pose_num
        self.patch_idx = patch_idx % self.edge_num

        # --- get poses and patch coords ---

        # get frozen poses (source and target), with no gradient to save unoptimized values as reference to optimization 
        with torch.no_grad():
            source_poses = self.poses[:, self.source_poses_idx, :].clone()
            target_poses = self.poses[:, self.target_poses_idx, :].clone()
            
            patch_coords = self.patch_coords[:, self.patch_idx, :]
        
            elevation_angle = self.elevation_angle[:, self.patch_idx].clone()
    
            # --- compose coords --- 
            source_coords = torch.cat([patch_coords, elevation_angle], dim = 2)
        
            # --- transform points --- 
            target_coords = self.transform(source_poses, target_poses, source_coords)
            
        # --- add corrections ---
        self.target_coords = target_coords[:, :, :2] + delta 
        
        # --- save initial state ---
        self.init_poses = pp.SE3(self.poses.tensor().clone().detach())
        self.init_elevation_angle = self.elevation_angle.clone().detach()

        # --- compose weights matrix --- 

        # weights refered to optimized parameters
        weights_param = weights.flatten()
        
        # prior/anchor 
        prior_weight = 1e-4

        weights_anchor_pose = torch.full((self.pose_num * 6,), prior_weight, device=weights.device, dtype=weights.dtype)
        weights_anchor_elev = torch.full((self.edge_num * 1,), prior_weight, device=weights.device, dtype=weights.dtype)
        
        weights = torch.cat([weights_param, weights_anchor_pose, weights_anchor_elev])
        self.weights = torch.diag(weights)

        
        # # --- set proper form o f weights 
        # self.weights = torch.diag(weights.flatten())

    def forward(self, dummy_input=None):

        # --- get poses and coords ---
        source_poses = self.poses[:, self.source_poses_idx, :]
        target_poses = self.poses[:, self.target_poses_idx, :]

        patch_coords = self.patch_coords[:, self.patch_idx % self.edge_num, :]
        elevation_angle = self.elevation_angle[:, self.patch_idx % self.edge_num, :]
        
        proj_coords = torch.cat([patch_coords, elevation_angle], dim = 2)

        # --- transfom coords ---
        proj_coords = self.transform(source_poses, target_poses, proj_coords)

        # --- projection error ---
        residual_proj = proj_coords[:, :, :2] - self.target_coords
        residual_proj = residual_proj.view(1, -1)
        # --- pose diff err --- 
        # print(f'init poses: {self.init_poses.shape}, act poses: {self.poses.shape}')
        residual_pose = (self.init_poses.Inv() @ self.poses).Log()
        residual_pose = residual_pose.view(1, -1)
        # --- elev ang err --- 
        residual_elev = self.elevation_angle - self.init_elevation_angle
        residual_elev = residual_elev.view(1, -1)
        # print('cat')
        # print(f'{residual_proj.shape}, {residual_pose.shape}, {residual_elev.shape}')
        residual = torch.cat([residual_proj, residual_pose, residual_elev], dim=1)

        return residual 


    def run(self, max_iter, early_stop_tol):
        
        optimizer = pp.optim.LM(self)
        prev_loss = float('inf')
        with torch.enable_grad():
            for i in range(max_iter):
            
                loss = optimizer.step(input = None, weight=self.weights)

                if abs(prev_loss - loss.item()) < early_stop_tol:
                        break
                    
                prev_loss = loss.item()
        
        return self.poses.tensor().view(self.b, self.n, 7), self.elevation_angle.view(self.b, self.n, self.p, 1)