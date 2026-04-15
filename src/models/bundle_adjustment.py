import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import transform_cart2polar, transform_polar2cart, depth_to_elev_angle

import pypose as pp

class BundleAdjustment(nn.Module):
    def __init__(self, supervised,
                 init_poses, 
                 init_patch_coords_r_theta, 
                 init_patch_coords_phi, 
                 source_frame_idx, target_frame_idx, patch_idx,
                 delta, weights,
                 sonar_param, freeze_poses):
        
        super().__init__()

        # --- init ---
        self.device = init_poses.device
        self.supervised = supervised
        self.sonar_param = sonar_param

        self.err_scale = torch.tensor([1.0, 0.1], device = self.device)

        if freeze_poses < 1:
            freeze_poses = 1
        self.freeze_poses = freeze_poses # not optimized poses number
        
        # physical to fls units scaling 
        self.physic2fls_scale_factor = torch.tensor([sonar_param.resolution.bins / (sonar_param.range.max - sonar_param.range.min),
                                                     sonar_param.resolution.beams / sonar_param.fov.horizontal], device = self.device).view(1, 1, 2)

        # save input shape:
        self.b, self.n_total, self.p, _ = init_patch_coords_r_theta.shape
        self.act_n = init_poses.shape[1]
        poses_n = self.b*self.act_n
        self.edges_n = self.b*self.act_n*self.p
        self.edges_total = self.b*self.n_total*self.p

        # global idx -> local idx
        self.source_frame_idx = source_frame_idx % poses_n
        self.target_frame_idx = target_frame_idx % poses_n
        self.patch_idx = patch_idx % self.edges_n

    
        # --- define not optimized parameters --- 

        # get patches coordinates
        self.patch_coords_r_theta = init_patch_coords_r_theta.view(1, self.edges_total, 2)
        
        # --- define parameters to optimize --- 

        # get initial poses to be optimized
        init_poses = init_poses.view(1, poses_n, 7)
        # normalize quaterions 
        init_poses[:, :, 3:] = F.normalize(init_poses[:, :, 3:], p=2, dim=-1)

        # define as pytorch/pypose parameters to optimize
        init_poses_se3 = pp.SE3(init_poses)

        if freeze_poses >= self.act_n:
            self.poses_anchor = init_poses_se3
            self.split_poses = False
        else:
            self.poses_anchor = init_poses_se3[:, :freeze_poses, :]
            self.translation_optim = nn.Parameter(init_poses[:, freeze_poses:, :3])
            self.rotation_optim = nn.Parameter(init_poses[:, freeze_poses:, 3:])
            # self.poses_optim = pp.Parameter(init_poses_se3[:, freeze_poses:, :])
            self.split_poses = True

        patch_coords_phi = init_patch_coords_phi.view(1, self.edges_total, 1)
        self.elevation_angle = nn.Parameter(patch_coords_phi) 


        # --- projection base line ---
                     
        # project points with act pose, add delta 
        source_poses = init_poses_se3[:, self.source_frame_idx, :].clone().detach()
        target_poses = init_poses_se3[:, self.target_frame_idx, :].clone().detach()
        # detach(), becouse nn.Parameter used to create these tensors

        patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :] 
        elevation_angle = self.elevation_angle[:, self.patch_idx].clone().detach() 
        source_coords = torch.cat([patch_coords, elevation_angle], dim = 2)
    
        target_coords = transform(source_poses, target_poses, source_coords)
            
        # projcted coords (r, theta) - baseline for BA optimization
        self.coords_baseline = target_coords[:, :, :2] * self.physic2fls_scale_factor + delta

        # weights for optimization
        self.weights = weights
       

    def forward(self, dummy_input=None):

        # compose pose tensor and coord tensor
        if self.split_poses:
            rotation_optim_norm = F.normalize(self.rotation_optim, p=2, dim=-1)
            poses_optim = torch.cat([self.translation_optim, rotation_optim_norm], dim=-1)
            poses_raw = torch.cat([self.poses_anchor, poses_optim], dim=1)
            poses = pp.SE3(poses_raw)
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
        
        # print(f'shape: {project_err.shape} * {self.weights.shape}')
        weighted_err = project_err.view(-1, 2) * self.err_scale * self.weights
        
        return weighted_err

    def run(self, max_iter, patience=10, min_delta = 1e-3, lr_elev=0.01, lr_rot=0.005, lr_trans=0.01, disp_stats=False):

        param_groups = [
            {'params': [self.elevation_angle], 'lr': lr_elev}
        ]
        
        if self.split_poses or self.freeze_poses == 0:
            param_groups.append({'params': [self.translation_optim], 'lr': lr_trans})
            param_groups.append({'params': [self.rotation_optim], 'lr': lr_rot})

        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_loss = float('inf')
        best_elev_angle = None
        best_pose_raw = None
        cntr = 0

        with torch.enable_grad(): 
            for i in range(max_iter):   
                optimizer.zero_grad()
                err = self.forward()
                
                loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=5.0)
                loss.backward()
                optimizer.step()
                current_loss = loss.item()
                scheduler.step(current_loss)

                if disp_stats:
                    print(f'Loss {i} iter: {current_loss:.4f} | r err: {err[:, 0].abs().mean().item():.4f} | theta err: {err[:, 1].abs().mean().item():.4f}')

                if current_loss + min_delta < best_loss:
                    best_loss = current_loss
                    cntr = 0
                    with torch.no_grad():
                        if self.split_poses:
                            best_translation = self.translation_optim.clone()
                            best_rotation = F.normalize(self.rotation_optim.clone(), p=2.0, dim=-1)
                            best_optim_part = torch.cat([best_translation, best_rotation], dim=-1)
                            
                            best_pose_raw = torch.cat([self.poses_anchor.tensor(), best_optim_part], dim=1)
                        else:
                            best_pose_raw = self.poses_anchor.tensor().clone()
                        
                        best_elev_angle = self.elevation_angle.clone()
                else:
                    cntr += 1
                    if cntr > patience:
                        break
        
        if best_pose_raw is None:
            if self.split_poses:
                rot_norm = F.normalize(self.rotation_optim.detach(), p=2.0, dim=-1)
                optim_part = torch.cat([self.translation_optim.detach(), rot_norm], dim=-1)
                best_pose_raw = torch.cat([self.poses_anchor.tensor(), optim_part], dim=1)
            else:
                best_pose_raw = self.poses_anchor.tensor()
            best_elev_angle = self.elevation_angle.detach()

        pose_optimized = best_pose_raw.detach().view(self.b, self.act_n, 7)
        pose_optimized[:, :, 3:] = F.normalize(pose_optimized[:, :, 3:], p=2, dim=1)
        elevation_optimized = best_elev_angle.detach().view(self.b, self.n_total, self.p, 1)

        return pose_optimized, elevation_optimized
    

    # def run(self, max_iter, patience=10, lr_elev=0.01, lr_rot=0.005, lr_trans = 0.01, disp_stats=False):

    #     param_groups = [
    #         {'params': [self.elevation_angle], 'lr': lr_elev}
    #     ]
        
    #     if self.split_poses or self.freeze_poses == 0:
    #         param_groups.append({'params': [self.translation_optim], 'lr': lr_trans})
    #         param_groups.append({'params': [self.rotation_optim], 'lr': lr_rot})

    #     optimizer = torch.optim.Adam(param_groups)

    #     best_loss = float('inf')
    #     best_elev_angle = None
    #     best_pose = None
    #     cntr = 0

    #     with torch.enable_grad(): 
    #         for i in range(max_iter):   
    #             optimizer.zero_grad()
    #             err = self.forward()
    #             # loss = (err ** 2).mean()
    #             loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=5.0)
    #             loss.backward()
    #             optimizer.step()

    #             # save if loss 
    #             if loss.item() < best_loss:
    #                 best_loss = loss.item()
    #                 cntr = 0
    #                 with torch.no_grad():
    #                     if self.split_poses:
    #                         best_translation = self.translation_optim.clone()
    #                         best_rotation = F.normalize(self.rotation_optim.clone(), p=2, dim=-1)
    #                         best_pose = torch.cat([best_translation, best_rotation], dim=-1)
    #                     else:
    #                         best_pose = self.poses_anchor.clone()
                        
    #                     best_elev_angle = self.elevation_angle.clone()
    #             else:
    #                 cntr += 1
    #                 if cntr > patience:
    #                     break

    #             if disp_stats:
    #                 print(f'Loss {i} iter: {loss.item()} | r err: {err[:, 0].abs().mean().item()} | theta err: {err[:, 1].abs().mean().item()}')
    #                 # print(f'loss {i} iter:', loss.item())
        
    #     # compose output tensor with optimized and non optimized poses
    #     if self.split_poses and not best_pose is None:
    #         pose_optimized = torch.cat([self.poses_anchor, best_pose], dim=1) 
    #         elevation_optimiezd = best_elev_angle
    #     else:
    #         pose_optimized = self.poses_anchor
    #         elevation_optimiezd = self.elevation_angle

    #     # detach from otim graph, restore orginal shape
    #     pose_optimized = pose_optimized.detach().view(self.b, self.act_n, 7)
    #     elevation_optimized = elevation_optimiezd.detach().view(self.b, self.n_total, self.p, 1)

    #     return pose_optimized, elevation_optimized
       

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
       
