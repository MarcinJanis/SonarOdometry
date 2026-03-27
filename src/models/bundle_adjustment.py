import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import transform_cart2polar, transform_polar2cart

import pypose as pp


class BundleAdjustment(nn.Module):
    def __init__(self, poses, patch_coords_r_theta, patch_coords_phi, sonar_param, freeze_poses = False):
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
        
        self.n_act = poses.shape[1]
        self.pose_num = self.b*self.n_act
        self.edge_num = self.b*self.n*self.p

        poses = poses.view(1, self.pose_num, 7)
        patch_coords_r_theta = patch_coords_r_theta.view(1, self.edge_num, 2)
        patch_coords_phi = patch_coords_phi.view(1, self.edge_num, 1)

        # --- define parameters to optimize ---
        poses_se3 = pp.SE3(poses)
        
        if freeze_poses:
            self.poses = poses_se3
        else:
            self.poses = pp.Parameter(poses_se3)
            
        self.elevation_angle = nn.Parameter(patch_coords_phi) # pp.Parameter(patch_coords_phi)

        # --- define constants parameters --- 
        self.patch_coords = patch_coords_r_theta
        self.sonar_param = sonar_param


    def transform(self, source_poses, target_poses, coords):

        source_poses = source_poses.squeeze(0)
        target_poses = target_poses.squeeze(0)
        coords = coords.squeeze(0)

        local_source_coords = transform_polar2cart(coords) # transorm_points_coords(coords, projection_type.POLAR2CARTESIAN)

        global_coords = source_poses @ local_source_coords

        local_target_coords = target_poses.Inv() @ global_coords

        coords = transform_cart2polar(local_target_coords)

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
        self.target_coords = target_coords[:, :, :2] + self.scale_delta(delta)
        
        # --- save initial state ---
        self.init_poses = pp.SE3(self.poses.tensor().clone().detach())
        self.init_elevation_angle = self.elevation_angle.clone().detach()

        # --- compose weights matrix --- 

        # weights refered to optimized parameters
        weights_param = weights.flatten()
        
        # prior/anchor 
        prior_weight = 1e-4

        weights_anchor_pose = torch.full((self.pose_num * 7,), prior_weight, device=weights.device, dtype=weights.dtype)

        weights_anchor_elev = torch.full((self.edge_num * 1,), prior_weight, device=weights.device, dtype=weights.dtype)
        
        weights = torch.cat([weights_param, weights_anchor_pose, weights_anchor_elev])
        self.weights = torch.diag(weights)
        
        # # --- set proper form o f weights 
        # self.weights = torch.diag(weights.flatten())

    # ===================================================================
    
    # ====================================================================
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
        residual_proj = self.scale_proj_err(residual_proj)
        residual_proj = residual_proj.view(1, -1)
        # --- pose diff err --- 
        
        # residual_pose = (self.init_poses.Inv() @ self.poses).Log().tensor() # numerical err in quaternions
        residual_pose = self.poses.tensor() - self.init_poses.tensor() # Safe option

        residual_pose = residual_pose.view(1, -1)
        # --- elev ang err --- 
        residual_elev = self.elevation_angle - self.init_elevation_angle
        residual_elev = residual_elev.view(1, -1)
        residual = torch.cat([residual_proj, residual_pose, residual_elev], dim=1)

        return residual 


    def run(self, max_iter, early_stop_tol=1e-3, trust_region=2.0):
        
        # optimizer = pp.optim.LM(self)
        strategy = pp.optim.strategy.TrustRegion(radius = trust_region) # define limits for optimized - damping, radius [m]
        optimizer = pp.optim.LM(self, strategy=strategy)

        prev_loss = float('inf')
        with torch.enable_grad():
            for i in range(max_iter):
            
                loss = optimizer.step(input = None, weight=self.weights)
                loss_diff = prev_loss - loss.item()
                if abs(loss_diff) < early_stop_tol:
                        print(f'[Bundle Adjustment Module] Early stopping')
                        break
                    
                prev_loss = loss.item()
        
        return self.poses.tensor().view(self.b, self.n_act, 7), self.elevation_angle.view(self.b, self.n, self.p, 1), loss_diff
    
    def scale_proj_err(self, proj_err):
        err_r = proj_err[:, :, 0] / (self.sonar_param.range.max - self.sonar_param.range.min) * self.sonar_param.resolution.bins
        err_t = proj_err[:, :, 1] / self.sonar_param.fov.horizontal * self.sonar_param.resolution.beams

        return torch.stack((err_r, err_t), dim=-1)

    def scale_delta(self, delta):
        delta_r = delta[:, 0] / self.sonar_param.resolution.bins * (self.sonar_param.range.max - self.sonar_param.range.min)
        delta_t = delta[:, 1] / self.sonar_param.resolution.beams * (self.sonar_param.fov.horizontal)
        return torch.stack((delta_r, delta_t), dim=-1)



