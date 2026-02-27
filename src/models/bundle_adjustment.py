import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import transorm_points_coords, projection_type, transform_matrix

import pypose as pp


class BundleAdjustment(nn.Module):
    def __init__(self, poses, patch_coords_r_theta, patch_coords_phi):
        super().__init__()
        
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

        pose_num = self.b*self.n
        edge_num = self.b*self.n*self.p

        poses = poses.view(1, pose_num, 7)
        patch_coords_r_theta = patch_coords_r_theta.view(1, edge_num, 2)
        patch_coords_phi = patch_coords_phi.view(1, edge_num, 1)

        # --- define parameters to optimize ---
        poses_se3 = pp.SE3(poses)
        self.poses = pp.Parameter(poses_se3)

        self.elevation_angle = nn.Parameter(patch_coords_phi) # pp.Parameter(patch_coords_phi)

        # --- define constants parameters --- 
        self.patch_coords = patch_coords_r_theta
        


    def transform(self, poses, coords):
        poses = poses.squeeze(0)
        coords = coords.squeeze(0)
        coords = transorm_points_coords(coords, projection_type.POLAR2CARTESIAN)
        coords = poses @ coords
        coords = transorm_points_coords(coords, projection_type.CARTESIAN2POLAR)
        poses = poses.unsqueeze(0)
        coords = coords.unsqueeze(0)
        return coords
    


    def init_ba(self, poses_idx, patch_idx, delta, weights):
        
        self.poses_idx = poses_idx
        self.patch_idx = patch_idx

        # --- get poses and patch coords ---
        poses = self.poses[:, self.poses_idx, :]
        
        patch_coords = self.patch_coords[:, self.patch_idx, :]
        
        elevation_angle = self.elevation_angle[:, self.patch_idx]
       
        self.weights = torch.diag(weights.flatten())
       
        
        with torch.no_grad():

            # --- compose coords --- 
            target_coords = torch.cat([patch_coords, elevation_angle], dim = 2)
            
            # --- transform points --- 
            
            target_coords = self.transform(poses, target_coords)
            
            # --- add corrections ---
            target_coords = target_coords[:, :, :2] +  delta 

            self.target_coords = target_coords.detach()
        

    def forward(self, dummy_input=None):

        # --- get poses and coords ---
        poses = self.poses[:, self.poses_idx, :]

        patch_coords = self.patch_coords[:, self.patch_idx, :]
        elevation_angle = self.elevation_angle[:, self.patch_idx, :]
        
        proj_coords = torch.cat([patch_coords, elevation_angle], dim = 2)

        # --- transfom coords ---
        proj_coords = self.transform(poses, proj_coords)

        # --- projection error ---
        residual = proj_coords[:, :, :2] - self.target_coords

        return residual 


    def run(self, max_iter, early_stop_tol):
        
        optimizer = pp.optim.LM(self)
        prev_loss = float('inf')
        
        for i in range(max_iter):
        
            loss = optimizer.step(input = None, weight=self.weights)
         
            if abs(prev_loss - loss.item()) < early_stop_tol:
                    break
                
            prev_loss = loss.item()

        optimized_poses = self.poses.detach()
        optimized_elevation = self.elevation_angle.detach()
        
        return optimized_poses.tensor().view(self.b, self.n, 7), optimized_elevation.view(self.b, self.n, self.p, 1)