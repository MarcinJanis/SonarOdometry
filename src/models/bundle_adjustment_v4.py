import torch
import torch.nn as nn
import pypose as pp
from .utils import project_points 

class BundleAdjustment(nn.Module):
    def __init__(self, 
                 init_poses, 
                 init_patch_coords_r_theta, init_patch_coords_phi, 
                 source_frame_idx, target_frame_idx, patch_idx,
                 delta, weights, freeze_poses, 
                 model_config, sonar_config, 
                 damping=False):

        super().__init__()
        self.device = init_poses.device
        
        self.freeze_poses = max(1, freeze_poses)
        
        self.r_min = sonar_config.range.min
        self.r_max = sonar_config.range.max
        self.fov_horizontal = sonar_config.fov.horizontal
        self.fls_h = model_config.FLS_INPUT_HEIGHT
        self.fls_w = model_config.FLS_INPUT_WIDTH

        self.b, self.n_total, self.p, _ = init_patch_coords_r_theta.shape
        self.act_n = init_poses.shape[1]
        poses_n = self.b * self.act_n
        
        self.s_idx = (source_frame_idx % poses_n).squeeze() # source frame local index
        self.t_idx = (target_frame_idx % poses_n).squeeze() # target frame local index 
        self.patch_idx = (patch_idx % (self.b * self.act_n * self.p)).squeeze() # patch local idx
        # note: work only becouse patch coors define for whole batch. If not, buff_size instead actual number of poses  
                     
        # poses
        self.init_poses_se3 = pp.SE3(init_poses)
        
        # POPRAWKA: Bezpośrednie użycie identity_SE3 z pypose i ujednolicenie wymiaru (act_n zamiast n_total)
        self.poses_correction = pp.Parameter(pp.identity_SE3(self.b, self.act_n - self.freeze_poses, device=self.device))

        # patch coords
        self.patch_coords_r_theta = init_patch_coords_r_theta
        self.patch_coords_phi = nn.Parameter(init_patch_coords_phi)              

        self.weights = weights 
        
        # projection base line
        self.projection_baseline = self._baseline(init_poses, delta)
                     
    def _get_opt_poses(self):
        freezed_poses = pp.identity_SE3(self.b, self.freeze_poses, device=self.device)
        poses_correction_full = torch.cat((freezed_poses, self.poses_correction.detach()), dim=1)
        
        return self.init_poses_se3 @ poses_correction_full
        
    def _baseline(self, poses, delta):
        elevation_angle = self.patch_coords_phi.detach().clone()
        
        flat_r_theta = self.patch_coords_r_theta.view(-1, 2)
        patch_coords_r_theta_expand = flat_r_theta[self.patch_idx] 
        
        flat_phi = elevation_angle.view(-1, 1)
        patch_coords_phi_expand = flat_phi[self.patch_idx]
        
        patch_coords = torch.cat((patch_coords_r_theta_expand, patch_coords_phi_expand), dim=-1)
        
        b, n, _ = poses.shape

        poses = poses.detach().clone()
        poses = poses.view(b*n, 7)
        
        source_poses = poses[self.s_idx]
        target_poses = poses[self.t_idx]

        target_pts = project_points(patch_coords, source_poses, target_poses)
        projection_baseline = scale_phisical2fls(target_pts[:, :2]) + delta
        return projection_baseline

    def get_projection_err(self):
        
        elevation_angle = self.patch_coords_phi.detach().clone()
        
        flat_r_theta = self.patch_coords_r_theta.view(-1, 2)
        patch_coords_r_theta_expand = flat_r_theta[self.patch_idx] 
        
        flat_phi = elevation_angle.view(-1, 1)
        patch_coords_phi_expand = flat_phi[self.patch_idx]
        
        patch_coords = torch.cat((patch_coords_r_theta_expand, patch_coords_phi_expand), dim=-1)
        
        b, n, _ = poses.shape
            
        poses_se3 = self._get_opt_poses()
        poses = poses_se3.tensor()
        
        source_poses = poses[self.s_idx]
        target_poses = poses[self.t_idx]

        target_pts = project_points(patch_coords, source_poses, target_poses)
        projection = scale_phisical2fls(target_pts[:, :2])

        err = self.projection_baseline - projection 
        return err

    

    
    def scale_phisical2fls(self, coords):
        # range r 
        r_norm = (coords[:, 0] - self.r_min) / (self.r_max - self.r_min)
        r = r_norm * self.fls_h

        # azimuth angle theta 
        theta_norm = coords[:, 1] / self.fov_horizontal 
        theta = (theta_norm + 0.5) * self.fls_w
        
        return torch.stack([r, theta], dim = 1)
        
