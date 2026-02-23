import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import transorm_points_coords, transform_matrix, projection_type

import pypose as pp


class BoundleAdjustment(nn.Module):
    def __init__(self, poses, patch_coords):
        super().__init__()

        poses_se3 = pp.SE3(poses)
        self.poses = pp.Parameter(poses_se3)
        
        self.patch_coords = patch_coords[:, :2]
        self.elevation_angle = pp.Parameter(patch_coords[:, 2:3])


    def init_ba(self, poses_idx, patch_idx, delta, weights):
        
        self.poses_idx = poses_idx
        self.patch_idx = patch_idx

        # --- get poses and patch coords ---
        poses = self.poses[self.poses_idx]
        patch_coords = self.patch_coords[self.patch_idx]
        elevation_angle = self.elevation_angle[self.patch_idx]

        self.weights = weights
        
        
        with torch.no_grad():

            # --- comppose coords --- 
            target_coords = torch.cat([patch_coords, elevation_angle], dim = 1)
            
            # --- transform points --- 
            T = transform_matrix(poses)
            target_coords  = T @ target_coords # złożyć target coords z tych dóch wcześniejsyzch 
            target_coords = transorm_points_coords(target_coords, projection_type.CARTESIAN2POLAR)
            
            # --- add corrections ---
            target_coords = target_coords + delta 
            self.target_coords = target_coords.detach()
        

    def forward(self):

        # --- get poses and coords ---
        poses = self.poses[self.poses_idx]
        
        patch_coords = self.patch_coords[self.patch_idx]
        elevation_angle = self.elevation_angle[self.patch_idx]
        
        proj_coords = torch.cat([patch_coords, elevation_angle], dim = 1)
        
        T = transform_matrix(poses)
        proj_coords  = T @ proj_coords
        proj_coords = transorm_points_coords(proj_coords, projection_type.CARTESIAN2POLAR)

        # calc projection error
        resiudal = (proj_coords - self.target_coords) * self.weights 
    
        return reisudal 

    def run(self, max_iter, early_stop_tol):
        
        optimizer = pp.optim.LM(self)
        
        prev_loss = float('inf')
        
        for i in range(max_iter):
            
            loss = optimizer.step()
            
            if abs(prev_loss - loss.item()) < early_stop_tol:
                    break
            prev_loss = loss.item()

        optimized_poses = self.poses.detach()
        optimized_elevation = self.elevation_angle.detach()
        
        return optimized_poses, optimized_elevation


