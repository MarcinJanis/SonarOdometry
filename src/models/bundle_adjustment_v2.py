import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import transform_cart2polar, transform_polar2cart, project_points

import pypose as pp

class BundleAdjustment(nn.Module):
    def __init__(self, 
                 init_poses, 
                 init_patch_coords_r_theta, 
                 init_patch_coords_phi, 
                 source_frame_idx, target_frame_idx, patch_idx,
                 delta, weights,
                 physic2fls_scale_factor, freeze_poses, 
                 damping = False):
        
        super().__init__()

        # --- init ---
        self.device = init_poses.device
        # self.sonar_param = sonar_param

        self.damping = damping


        # For details see BA_test.ipynb
        # damping_trans_weight = scale [pix/m] / (2 * max_permissible_trans)
        # damping_trans_weight = scale [pix/rad] / (2 * max_permissible_rot)
        self.damping_trans_weight = 2.63 / 50 # 9.329494828396749 
        self.damping_rot_weight = 356.51 / 50# 2608.996360840966 

        if freeze_poses < 1:
            freeze_poses = 1
        self.freeze_poses = freeze_poses # not optimized poses number
        
        # physical to fls units scaling 

        self.physic2fls_scale_factor = physic2fls_scale_factor

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

        self.patch_coords_r_theta = init_patch_coords_r_theta.view(1, self.edges_total, 2)
        
        # get initial poses as optimization base, saved as SE3 objects
        init_poses = init_poses.clone()
        init_poses[:, :, 3:] = F.normalize(init_poses[:, :, 3:], p=2, dim=-1) # normalize quaterions 
        self.init_poses_se3 = pp.SE3(init_poses)

        # --- define parameters to optimize --- 

        # define optimize parameters
        if freeze_poses >= self.act_n:
            self.optimize_poses = False
        else:
            self.optimize_poses = True
            num_optim = self.act_n - self.freeze_poses # number os poses to be optimized

            # translation and rotation correction - optimized parameters
            self.trans_correction = nn.Parameter(torch.zeros(self.b, num_optim, 3, device=self.device))
            self.rot_correction = nn.Parameter(torch.zeros(self.b, num_optim, 3, device=self.device))
        
        # phi angle - optimized parameter
        patch_coords_phi = init_patch_coords_phi.view(1, self.edges_total, 1)
        self.elevation_angle = nn.Parameter(patch_coords_phi) 


        # --- optimization baseline ---
        # project points with actual poses, add delta from net. 
        # Result will be baseline for poses and phi angle optimization

        # reproject points with act pose
        init_poses_flat = init_poses.view(1, poses_n, 7)
        source_poses = init_poses_flat[:, self.source_frame_idx, :].clone()
        target_poses = init_poses_flat[:, self.target_frame_idx, :].clone()
        
        patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :] 
        elevation_angle = self.elevation_angle[:, self.patch_idx].clone().detach() 
        source_coords = torch.cat([patch_coords, elevation_angle], dim = 2)
    
        target_coords = project_points(source_coords.squeeze(0), source_poses.squeeze(0), target_poses.squeeze(0))
        target_coords = target_coords.unsqueeze(0)

        # rescale reprojected points to fls/pixel units, add delta
        self.coords_baseline = target_coords[:, :, :2] * self.physic2fls_scale_factor + delta

        # weights for optimization
        self.weights = weights
       

    def get_pose_estim(self):
        '''
        Get initial poses, add optimized correction
        '''
        if not self.optimize_poses:
                return self.init_poses_se3
        else:
            # connect trans and rotat
            poses_correction = torch.cat((self.trans_correction, self.rot_correction), dim=-1)
            # set as se3 (6d) object and transform to SE3 (7d)
            poses_correction_SE3 = pp.se3(poses_correction).Exp()
            
            # add corrections
            base_poses = self.init_poses_se3[:, self.freeze_poses:, :] # poses to change
            new_poses_se3 = base_poses @ poses_correction_SE3 # changed poses 

            frozen_poses = self.init_poses_se3[:, :self.freeze_poses, :] # unchanged poses
            return torch.cat([frozen_poses, new_poses_se3], dim=1)


    def forward(self, dummy_input=None):

        # get optimized poses
        poses = self.get_pose_estim()
        
        # get optimized elevation angle, add to r and theta coords
        patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :]
        elevation_angle = self.elevation_angle[:, self.patch_idx, :] # pp.Parameter
        source_coords = torch.cat([patch_coords, elevation_angle], dim = -1)

        # --- reproject points with actual poses --- 

        # expand for all edges
        poses_flat = poses.view(1, self.b * self.act_n, 7)
        source_poses = poses_flat[:, self.source_frame_idx, :]
        target_poses = poses_flat[:, self.target_frame_idx, :]

        # reproject
        projected_coords = project_points(source_coords.squeeze(0), 
                                          source_poses.squeeze(0).tensor(), 
                                          target_poses.squeeze(0).tensor())
        projected_coords = projected_coords.unsqueeze(0)

        # --- projection error ---
        
        # reprojection err - r, distance 
        projection_err_r = projected_coords[:, :, 0] * self.physic2fls_scale_factor[:, :, 0] - self.coords_baseline[:, :, 0]
        
        # reprojection err - theta, azimuth 
        # atan2 to forced projection err to (-pi, pi) range
        projection_err_theta_raw = projected_coords[:, :, 1] - (self.coords_baseline[:, :, 1] / self.physic2fls_scale_factor[:, :, 1])
        projection_err_theta_raw = torch.atan2(torch.sin(projection_err_theta_raw), 
                                               torch.cos(projection_err_theta_raw))

        projection_err_theta = projection_err_theta_raw * self.physic2fls_scale_factor[:, :, 1]
        
        project_err = torch.stack([projection_err_r, projection_err_theta], dim=2)
        
        # add weights, err scale
        weighted_err = project_err * self.weights
        
        return weighted_err

    def run(self, max_iter, patience=10, min_delta = 1e-3, lr_elev=0.01, lr_rot=0.005, lr_trans=0.01, disp_stats=False):

        # set learning rates for each parameters
        param_groups = [
            {'params': [self.elevation_angle], 'lr': lr_elev}
        ]

        if self.optimize_poses or self.freeze_poses == 0:
            param_groups.append({'params': [self.trans_correction], 'lr': lr_trans})
            param_groups.append({'params': [self.rot_correction], 'lr': lr_rot})

        optimizer = torch.optim.Adam(param_groups)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        best_loss = float('inf')
        best_elev_angle = None
        best_delta_pose = None

        cntr = 0

        with torch.enable_grad(): 
            for i in range(max_iter):   
                optimizer.zero_grad()
                err = self.forward()
                
                # --- smooth L1 loss ---
                #  (L1 for big err, L2 for small err)
                loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=10.0)

                # --- Add prior/damper for optimizer ---
                
                # to force back from too big changes, 
                # add to loss punishment, proportional to changed distance
                if self.damping and self.optimize_poses: 
                    prior_trans = torch.mean(self.trans_correction**2) * self.damping_trans_weight 
                    priot_rot = torch.mean(self.rot_correction**2) * self.damping_rot_weight 

                    loss = loss + prior_trans + priot_rot

                loss.backward()
                optimizer.step()
                current_loss = loss.item()
                # scheduler.step(current_loss)
                scheduler.step()

                if disp_stats:

                    r_err_mean = err[:, :, 0].abs().mean().item()
                    theta_err_mean = err[:, :, 1].abs().mean().item()
                    print(f'Loss {i} iter: {current_loss:.4f} | r err: {r_err_mean:.4f} | theta err: {theta_err_mean:.4f}')

                    # print(f'Loss {i} iter: {current_loss:.4f} | r err: {err[:, 0].abs().mean().item():.4f} | theta err: {err[:, 1].abs().mean().item():.4f}')

                if current_loss + min_delta < best_loss:
                    best_loss = current_loss
                    cntr = 0
                    with torch.no_grad():
                        if self.optimize_poses:
                            best_delta_pose = torch.cat([self.trans_correction.clone(), self.rot_correction.clone()], dim=-1)
                        best_elev_angle = self.elevation_angle.clone()
                else:
                    cntr += 1
                    if cntr > patience:
                        break
        
        if best_delta_pose is None:
            if self.optimize_poses:
                best_delta_pose = torch.cat([self.trans_correction, self.rot_correction], dim=-1)
            best_elev_angle = self.elevation_angle

        # post processing optimized values

        elevation_optimized = best_elev_angle.detach().view(self.b, self.n_total, self.p, 1)
        if self.optimize_poses:
            best_delta_poses_se3 = pp.se3(best_delta_pose.detach()).Exp()
            base_poses = self.init_poses_se3[:, self.freeze_poses:, :]
            new_poses_se3 = base_poses @ best_delta_poses_se3
            frozen_poses = self.init_poses_se3[:, :self.freeze_poses, :]

            # pose_optimized = torch.cat([frozen_poses, new_poses_se3], dim=1)
            pose_optimized = torch.cat([frozen_poses.tensor(), new_poses_se3.tensor()], dim=1)
        else:
            pose_optimized = self.init_poses_se3.tensor()

        return pose_optimized, elevation_optimized
    