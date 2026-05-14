import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import transform_cart2polar, transform_polar2cart, project_points

import pypose as pp
from pypose.optim import LM
from pypose.optim.strategy import TrustRegion
from pypose.optim.kernel import Huber

class BundleAdjustment(nn.Module):
    def __init__(self, 
                 init_poses, 
                 init_patch_coords_r_theta, init_patch_coords_phi, 
                 source_frame_idx, target_frame_idx, patch_idx,
                 delta, weights, freeze_poses, 
                 model_config, sonar_config, 
                 damping=False):
        
        super().__init__()

        # --- init ---
        self.device = init_poses.device
        self.damping = damping

        self.damping_trans_weight = 2.63 / 50 
        self.damping_rot_weight = 356.51 / 50 

        # freeze_poses - not optimized poses number
        if freeze_poses < 1:
            freeze_poses = 1
        self.freeze_poses = freeze_poses 
        
        # physical to fls units scaling 
        self.r_min = sonar_config.range.min
        self.r_max = sonar_config.range.max
        self.fov_horizontal = sonar_config.fov.horizontal
        self.fls_h = model_config.FLS_INPUT_HEIGHT
        self.fls_w = model_config.FLS_INPUT_WIDTH

        # save input shape:
        self.b, self.n_total, self.p, _ = init_patch_coords_r_theta.shape
        self.act_n = init_poses.shape[1]
        poses_n = self.b * self.act_n
        self.edges_n = self.b * self.act_n * self.p
        self.edges_total = self.b * self.n_total * self.p

        # global idx -> local idx
        self.source_frame_idx = source_frame_idx % poses_n
        self.target_frame_idx = target_frame_idx % poses_n
        self.patch_idx = patch_idx % self.edges_n

        # --- define not optimized parameters --- 
        self.patch_coords_r_theta = init_patch_coords_r_theta.view(1, self.edges_total, 2)
        
        # get initial poses as optimization base, saved as SE3 objects
        init_poses = init_poses.clone()
        init_poses[:, :, 3:] = F.normalize(init_poses[:, :, 3:], p=2, dim=-1) # normalize quaternions 
        self.init_poses_se3 = pp.SE3(init_poses)

        # --- define parameters to optimize --- 
        if freeze_poses >= self.act_n:
            self.optimize_poses = False
        else:
            self.optimize_poses = True
            self.num_optim = self.act_n - self.freeze_poses 
            
            # [ZMIANA] Jednolity parametr LieTensor dla translacji i rotacji (znacznie szybsze dla LM)
            # 6 wymiarów przestrzeni tangencjalnej se(3)
            self.pose_correction = pp.Parameter(pp.se3(torch.zeros(self.b, self.num_optim, 6, device=self.device)))
        
        # [ZMIANA] Rejestracja kąta jako pp.Parameter 
        # patch_coords_phi = init_patch_coords_phi.view(1, self.edges_total, 1)
        # self.elevation_angle = pp.Parameter(patch_coords_phi) 
        patch_coords_phi = init_patch_coords_phi.view(1, self.edges_total, 1)
        self.elevation_angle = nn.Parameter(patch_coords_phi)

        # --- optimization baseline ---
        init_poses_flat = init_poses.view(1, poses_n, 7)
        source_poses = init_poses_flat[:, self.source_frame_idx, :].clone()
        target_poses = init_poses_flat[:, self.target_frame_idx, :].clone()
        
        patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :] 
        elevation_angle = self.elevation_angle[:, self.patch_idx].clone().detach() 
        source_coords = torch.cat([patch_coords, elevation_angle], dim=2)
    
        target_coords = project_points(source_coords.squeeze(0), source_poses.squeeze(0), target_poses.squeeze(0))

        self.coords_baseline = self.scale_phisical2fls(target_coords[:, :2]) + delta
        self.weights = weights

    def get_pose_estim(self):
        '''
        Get initial poses, add optimized correction
        '''
        if not self.optimize_poses:
            return self.init_poses_se3
        else:
            # [ZMIANA] Mapowanie eksponencjalne bezpośrednio z LieTensora
            poses_correction_SE3 = self.pose_correction.Exp()
            
            base_poses = self.init_poses_se3[:, self.freeze_poses:, :] 
            new_poses_se3 = base_poses @ poses_correction_SE3 

            frozen_poses = self.init_poses_se3[:, :self.freeze_poses, :] 
            return torch.cat([frozen_poses, new_poses_se3], dim=1)

    def forward(self, dummy_input=None):
        # get optimized poses
        poses = self.get_pose_estim()
        
        # get optimized elevation angle, add to r and theta coords
        patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :]
        elevation_angle = self.elevation_angle[:, self.patch_idx, :] 
        source_coords = torch.cat([patch_coords, elevation_angle], dim=-1)

        # --- reproject points with actual poses --- 
        poses_flat = poses.view(1, self.b * self.act_n, 7)
        source_poses = poses_flat[:, self.source_frame_idx, :]
        target_poses = poses_flat[:, self.target_frame_idx, :]

        projected_coords = project_points(source_coords.squeeze(0), 
                                          source_poses.squeeze(0).tensor(), 
                                          target_poses.squeeze(0).tensor())
        
        # --- projection error ---
        projected_coords_fls = self.scale_phisical2fls(projected_coords)
        project_err = self.coords_baseline - projected_coords_fls 
        
        weighted_err = project_err * self.weights
        
        # [ZMIANA] LM oczekuje zwrotu surowych (nieskorelowanych kwadratowo) reszt. 
        # Spłaszczamy je do 1D.
        residuals = weighted_err.view(-1)

        # [ZMIANA] Damping w optymalizatorach LM realizuje się poprzez "sztuczne" błędy a priori.
        # Minimalizujemy: Sum(err^2) + Sum((x * w_damped)^2)
        if self.damping and self.optimize_poses: 
            # Normalizujemy wagi tłumienia podobnie jak w torch.mean(x**2)
            N_trans = self.b * self.num_optim * 3
            N_rot = self.b * self.num_optim * 3

            trans_corr = self.pose_correction[..., :3]
            rot_corr = self.pose_correction[..., 3:]

            # Pierwiastkujemy wagę, ponieważ optymalizator LM podniesie tę resztę do kwadratu
            prior_trans_res = trans_corr * math.sqrt(self.damping_trans_weight / N_trans)
            prior_rot_res = rot_corr * math.sqrt(self.damping_rot_weight / N_rot)

            # Doklejamy kary jako dodatkowe reszty
            residuals = torch.cat([residuals, prior_trans_res.view(-1), prior_rot_res.view(-1)])
            
        return residuals

    def run(self, max_iter, patience=10, min_delta=1e-3, lr_elev=None, lr_rot=None, lr_trans=None, disp_stats=False):
        # LRs nie są już potrzebne - zostawiłem je w sygnaturze dla kompatybilności 
        # ze starym kodem wywołującym (aby nic nie popsuć), ale nie wpływają na LM.

        # [ZMIANA] Inicjalizacja optymalizatora PyPose Levenberg-Marquardt
        # Używamy strategii TrustRegion - najstabilniejszej dla Bundle Adjustment.
        # Kernel Huber(delta=2.5) zachowa się identycznie jak poprzedni Smooth L1.
        strategy = TrustRegion()
        optimizer = LM(self, strategy=strategy, kernel=Huber(delta=2.5))

        best_loss = float('inf')
        best_elev_angle = None
        best_delta_pose = None
        cntr = 0

        for i in range(max_iter):   
            optimizer.zero_grad()
            
            # optimizer.step() automatycznie przelicza forward() oraz Jakobiany
            loss = optimizer.step(input=None)
            current_loss = loss.item()

            if disp_stats:
                print(f'Loss {i} iter: {current_loss:.4f}')

            if current_loss + min_delta < best_loss:
                best_loss = current_loss
                cntr = 0
                with torch.no_grad():
                    if self.optimize_poses:
                        best_delta_pose = self.pose_correction.clone()
                    best_elev_angle = self.elevation_angle.clone()
            else:
                cntr += 1
                if cntr > patience:
                    break
        
        if best_delta_pose is None:
            if self.optimize_poses:
                best_delta_pose = self.pose_correction.detach()
            best_elev_angle = self.elevation_angle.detach()

        # Post-processing
        elevation_optimized = best_elev_angle.detach().view(self.b, self.n_total, self.p, 1)
        
        if self.optimize_poses:
            # Ponownie używamy Exp() na zapisanym LieTensorze
            best_delta_poses_se3 = best_delta_pose.detach().Exp()
            base_poses = self.init_poses_se3[:, self.freeze_poses:, :]
            new_poses_se3 = base_poses @ best_delta_poses_se3
            frozen_poses = self.init_poses_se3[:, :self.freeze_poses, :]

            pose_optimized = torch.cat([frozen_poses.tensor(), new_poses_se3.tensor()], dim=1)
        else:
            pose_optimized = self.init_poses_se3.tensor()

        return pose_optimized, elevation_optimized

    def scale_fls2phisical(self, coords):
        # range r 
        r_norm = coords[:, 0] / self.fls_h
        r = r_norm * (self.r_max - self.r_min) + self.r_min

        # azimuth angle theta 
        theta_norm = coords[:, 1] / self.fls_w - 0.5
        theta = theta_norm * self.fov_horizontal 
        
        return torch.stack([r, theta], dim = 1)

    def scale_phisical2fls(self, coords):
        # range r 
        r_norm = (coords[:, 0] - self.r_min) / (self.r_max - self.r_min)
        r = r_norm * self.fls_h

        # azimuth angle theta 
        theta_norm = coords[:, 1] / self.fov_horizontal 
        theta = (theta_norm + 0.5) * self.fls_w
        
        return torch.stack([r, theta], dim = 1)