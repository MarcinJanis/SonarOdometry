import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import transform_cart2polar, transform_polar2cart, depth_to_elev_angle

import pypose as pp

class BundleAdjustment(nn.Module):
    def __init__(self, supervised,
                 init_poses, 
                 gt_poses, gt_depth,
                 init_patch_coords_r_theta, 
                 init_patch_coords_phi, 
                 source_frame_idx, target_frame_idx, patch_idx,
                 delta, weights,
                 sonar_param, freeze_poses):
        
        super().__init__()
        self.device = init_poses.device
        self.supervised = supervised

        # --- init ---
        if freeze_poses < 1:
            freeze_poses = 1
        self.freeze_poses = freeze_poses
        
        # physical to fls units scaling 
        self.physic2fls_scale_factor = torch.tensor([sonar_param.resolution.bins / (sonar_param.range.max - sonar_param.range.min),
                                                     sonar_param.resolution.beams / sonar_param.fov.horizontal], device = self.device).view(1, 1, 2)


        # remember input shape:
        self.b, self.n_total, self.p, _ = init_patch_coords_r_theta.shape
        self.act_n = init_poses.shape[1]
        poses_n = self.b*self.act_n
        self.edges_n = self.b*self.act_n*self.p
        self.edges_total = self.b*self.n_total*self.p

        # get actual number of estimated poses 
        init_poses = init_poses.view(1, poses_n, 7)
        init_poses = _quat_norm(init_poses)

        # get acutal number of patch coords
        patch_coords_r_theta = init_patch_coords_r_theta.view(1, self.edges_total, 2)
        patch_coords_phi = init_patch_coords_phi.view(1, self.edges_total, 1)
        
        # --- define parameters to optimize ---

        init_poses_se3 = pp.SE3(init_poses)

        if freeze_poses >= self.act_n:
            self.poses_anchor = init_poses_se3
            self.split_poses = False
        else:
            self.poses_anchor = init_poses_se3[:, :freeze_poses, :]
            self.poses_optim = pp.Parameter(init_poses_se3[:, freeze_poses:, :])
            self.split_poses = True
            
        self.elevation_angle = nn.Parameter(patch_coords_phi) 

        # --- define parameters not optimized --- 
        self.patch_coords_r_theta = patch_coords_r_theta
        self.sonar_param = sonar_param

        # global idx -> local idx
        self.source_frame_idx = source_frame_idx % poses_n
        self.target_frame_idx = target_frame_idx % poses_n
        self.patch_idx = patch_idx % self.edges_n

        # --- projection base line ---
                     
        # project points with act pose, add delta
        source_poses = init_poses_se3[:, self.source_frame_idx, :].clone() # [clone() or detach() also!?]
        target_poses = init_poses_se3[:, self.target_frame_idx, :].clone() # [clone() or detach() also!?]
        
        patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :] 
        elevation_angle = self.elevation_angle[:, self.patch_idx].clone() # [clone() or detach() also!?]
        source_coords = torch.cat([patch_coords, elevation_angle], dim = 2)
    
        target_coords = transform(source_poses, target_poses, source_coords)
            
        # projcted coords (r, theta) - baseline for BA optimization
        self.coords_baseline = target_coords[:, :, :2] * self.physic2fls_scale_factor + delta

        # weights for optimization
        self.weights = weights.flatten()
       

    def forward(self, dummy_input=None):

        # compose pose tensor and coord tensor
        if self.split_poses:
            poses = torch.cat([self.poses_anchor, self.poses_optim], dim=1)
        else:
            poses = self.poses_anchor
        
        patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :]
        elevation_angle = self.elevation_angle[:, self.patch_idx, :] # pp.Parameter
        source_coords = torch.cat([patch_coords, elevation_angle], dim = 2)

        # expand for all edges
        source_poses = poses[:, self.source_frame_idx, :]
        target_poses = poses[:, self.target_frame_idx, :]

        # --- project --- 
        projected_coords = transform(source_poses, target_poses, source_coords)

        # --- projection error ---
        project_err = (projected_coords[:, :, :2] * self.physic2fls_scale_factor - self.coords_baseline) 
        
        weighted_err = project_err.flatten() * self.weights

        return weighted_err


    def run(self, max_iter, lr = 0.1, disp_stats=False) #early_stop_tol=1e-4, trust_region=2.0):

        params_to_opt = [self.elevation_angle]
        if self.split_poses:
            params_to_opt.append(self.poses_optim)
        elif self.freeze_poses == 0:
            params_to_opt.append(self.poses_anchor)

        optimizer = torch.optim.Adam(params_to_opt, lr=lr)

        if disp_stats:
            loss_iter = []
            
        with torch.enable_grad(): 
            for i in range(max_iter):   
                optimizer.zero_grad()
                err = self.forward()
                loss = (err ** 2).sum()
                loss.backward()
                optimizer.step()
                
                # norm quaterions after each optimization
                with torch.no_grad():
                    if self.split_poses:
                        self.poses_optim.data[:, :, 3:] = F.normalize(self.poses_optim.data[:, :, 3:], p=2, dim=-1)
                   
                self.poses_optim = _quat_norm(self.poses_optim) 
                
                if disp_stats:
                    print(f'loss {i} iter:', loss.item())
        
        if self.split_poses:
            poses_optimized_se3 = torch.cat([self.poses_anchor, self.poses_optim], dim=1)
        else:
            poses_optimized_se3 = self.poses_anchor

        pose_optimized = pose_optimized_se3.tensor().detach().view(self.b, self.act_n, 7)
        elevation_optimized = self.elevation_angle.detach().view(self.b, self.n_total, self.p, 1)

        return pose_optimized, elevation_optimized
       

def transform(source_poses, target_poses, coords):
    # project points from source pose to target pose frame of refernce
    # function operates on pp.SE3() objects

    source_poses = source_poses.squeeze(0)
    target_poses = target_poses.squeeze(0)
    coords = coords.squeeze(0)

    local_source_coords = transform_polar2cart(coords) 

    global_coords = source_poses @ local_source_coords

    local_target_coords = target_poses.Inv() @ global_coords

    coords = transform_cart2polar(local_target_coords)

    return coords.unsqueeze(0)

def _quat_norm(pose):
    pose[:, :, 3:] = F.normalize(pose[:, :, 3:], p=2, dim=-1)
    return pose
    
# def _quat_norm(pose):
#     t = pose[:, :, :3]
#     q = pose[:, :, 3:]
#     pose = torch.cat([t, F.normalize(q, ord=2.0, dim=-1)], dim=-1)
#     return pose
       
