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
                 sonar_param, freeze_poses):
        
        super().__init__()

        # --- INIT ---
        self.device = init_poses.device
        self.sonar_param = sonar_param

        # Wyrównane wagi błędów (możesz dostroić, ale 1.0 do radianów zawiniętych jest bezpieczne)
        self.err_scale = torch.tensor([1.0, 1.0], device=self.device)

        # Minimum 1 klatka zamrożona dla stabilności grafu
        self.freeze_poses = max(1, freeze_poses)
        
        # Współczynniki skalujące (jednostki fizyczne -> jednostki FLS)
        self.physic2fls_scale_factor = torch.tensor([
            sonar_param.resolution.bins / (sonar_param.range.max - sonar_param.range.min),
            sonar_param.resolution.beams / sonar_param.fov.horizontal
        ], device=self.device).view(1, 1, 2)

        # Zapisz wymiary
        self.b, self.n_total, self.p, _ = init_patch_coords_r_theta.shape
        self.act_n = init_poses.shape[1]
        poses_n = self.b * self.act_n
        self.edges_n = self.b * self.act_n * self.p
        self.edges_total = self.b * self.n_total * self.p

        # Indeksy lokalne krawędzi
        self.source_frame_idx = source_frame_idx % poses_n
        self.target_frame_idx = target_frame_idx % poses_n
        self.patch_idx = patch_idx % self.edges_n

        self.patch_coords_r_theta = init_patch_coords_r_theta.view(1, self.edges_total, 2)
        
        # --- OPTYMALIZOWANE PARAMETRY POZ --- 
        init_poses = init_poses.view(1, poses_n, 7)
        # Normalizacja kwaternionów
        init_poses[:, :, 3:] = F.normalize(init_poses[:, :, 3:], p=2, dim=-1)
        
        # Prawidłowy obiekt SE(3) rzutujący na algebry
        self.init_poses_se3 = pp.SE3(init_poses) 

        if self.freeze_poses >= self.act_n:
            self.optimize_poses = False
        else:
            self.optimize_poses = True
            num_optim = self.act_n - self.freeze_poses
            # Delty startują od czystego zera (Tangent Space)
            self.delta_trans = nn.Parameter(torch.zeros(self.b, num_optim, 3, device=self.device))
            self.delta_rot = nn.Parameter(torch.zeros(self.b, num_optim, 3, device=self.device))
        
        # --- OPTYMALIZOWANA ELEWACJA ---
        patch_coords_phi = init_patch_coords_phi.view(1, self.edges_total, 1)
        self.elevation_angle = nn.Parameter(patch_coords_phi) 

        # --- BAZA RZUTOWANIA (BASELINE) ---
        with torch.no_grad():
            source_poses = init_poses[:, self.source_frame_idx, :]
            target_poses = init_poses[:, self.target_frame_idx, :]

            patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :] 
            elevation_angle = self.elevation_angle[:, self.patch_idx]
            source_coords = torch.cat([patch_coords, elevation_angle], dim=2)
        
            # Przekazujemy CZYSTE tensory do utilsów
            target_coords = project_points(source_coords.squeeze(0), 
                                           source_poses.squeeze(0), 
                                           target_poses.squeeze(0)).unsqueeze(0)

            # Rzutujemy cel na piksele i dodajemy delta od Gru
            self.coords_baseline = target_coords[:, :, :2] * self.physic2fls_scale_factor + delta
            self.weights = weights
       
    def get_pose_estim(self):
        if not self.optimize_poses:
            return self.init_poses_se3.tensor()
            
        # Transformujemy Delty do Lie Manifold
        delta_poses = torch.cat((self.delta_trans, self.delta_rot), dim=-1)
        delta_poses_se3 = pp.se3(delta_poses).Exp()
        
        # Kompozycja operatorów Lie (odcięta baza + uaktualnienie)
        base_poses = self.init_poses_se3[:, self.freeze_poses:, :]
        new_poses_se3 = base_poses @ delta_poses_se3

        frozen_poses = self.init_poses_se3[:, :self.freeze_poses, :]
        # Dekodowanie do czystych tensorów
        return torch.cat([frozen_poses.tensor(), new_poses_se3.tensor()], dim=1)

    def forward(self, dummy_input=None):
        # Pobieranie TENSORA, nie obiektu PyPose
        poses = self.get_pose_estim()
        
        patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :]
        elevation_angle = self.elevation_angle[:, self.patch_idx, :]
        source_coords = torch.cat([patch_coords, elevation_angle], dim=2)

        source_poses = poses[:, self.source_frame_idx, :]
        target_poses = poses[:, self.target_frame_idx, :]

        # --- PROJEKCJA --- 
        projected_coords = project_points(source_coords.squeeze(0), 
                                          source_poses.squeeze(0), 
                                          target_poses.squeeze(0)).unsqueeze(0)

        # --- OBLICZANIE BŁĘDU ---
        # 1. BŁĄD R (Zasięg) w pixelach FLS
        project_err_r = projected_coords[:, :, 0] * self.physic2fls_scale_factor[:, :, 0] - self.coords_baseline[:, :, 0]
        
        # 2. BŁĄD THETA z ZAWIJANIEM radianów
        theta_proj_rad = projected_coords[:, :, 1]
        theta_base_rad = self.coords_baseline[:, :, 1] / self.physic2fls_scale_factor[:, :, 1]
        
        theta_diff_rad = theta_proj_rad - theta_base_rad
        theta_wrapped_rad = torch.atan2(torch.sin(theta_diff_rad), torch.cos(theta_diff_rad))
        
        # Skalowanie na FLS
        project_err_theta = theta_wrapped_rad * self.physic2fls_scale_factor[:, :, 1]
        
        # 3. BEZPIECZNE SKŁADANIE BŁĘDU (Gwarantuje poprawny shape bez wymieszania r i theta)
        project_err = torch.stack([project_err_r, project_err_theta], dim=2)
        
        weighted_err = project_err.view(-1, 2) * self.err_scale * self.weights
        return weighted_err

    def run(self, max_iter, patience=10, min_delta=1e-4, lr_elev=0.005, lr_rot=0.001, lr_trans=0.005, disp_stats=False):

        param_groups = [
            {'params': [self.elevation_angle], 'lr': lr_elev}
        ]
        
        if self.optimize_poses:
            param_groups.append({'params': [self.delta_trans], 'lr': lr_trans})
            param_groups.append({'params': [self.delta_rot], 'lr': lr_rot})

        # Zmniejszone Momentum (0.5), aby uniknąć przestrzeliwania na wygładzonym torze
        optimizer = torch.optim.Adam(param_groups, betas=(0.5, 0.999))
        
        # Zdecydowane schładzanie kroku - absolutny mus
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

        best_loss = float('inf')
        best_elev_angle = None
        best_delta_pose = None
        cntr = 0

        # WAGI GORSETU (Prior Loss) - wymuszają dociąganie predykcji, blokują losowe skakanie
        reg_weight_trans = 1.0 
        reg_weight_rot = 10.0  

        with torch.enable_grad(): 
            for i in range(max_iter):   
                optimizer.zero_grad()
                err = self.forward()
                
                # Błąd Rzutowania Obrazu
                data_loss = F.smooth_l1_loss(err, torch.zeros_like(err), beta=1.0)
                
                if self.optimize_poses:
                    # Kara za wariowanie parametrami (Odchylenie od predykcji GRU/Kinetmatyki)
                    prior_loss_trans = torch.sum(self.delta_trans ** 2) * reg_weight_trans
                    prior_loss_rot = torch.sum(self.delta_rot ** 2) * reg_weight_rot
                    loss = data_loss + prior_loss_trans + prior_loss_rot
                else:
                    loss = data_loss

                loss.backward()
                optimizer.step()
                # Aktualizacja Schedulera (dla Exponential bez argumentu current_loss)
                scheduler.step()

                current_loss = loss.item()

                if disp_stats:
                    print(f'Loss {i:02d}: {current_loss:.4f} | R err: {err[:, 0].abs().mean().item():.4f} | Theta err: {err[:, 1].abs().mean().item():.4f}')

                if current_loss + min_delta < best_loss:
                    best_loss = current_loss
                    cntr = 0
                    with torch.no_grad():
                        if self.optimize_poses:
                            best_delta_pose = torch.cat([self.delta_trans.clone(), self.delta_rot.clone()], dim=-1)
                        best_elev_angle = self.elevation_angle.clone()
                else:
                    cntr += 1
                    if cntr > patience:
                        break
        
        if best_elev_angle is None:
            best_elev_angle = self.elevation_angle
            
        if self.optimize_poses and best_delta_pose is None:
            best_delta_pose = torch.cat([self.delta_trans, self.delta_rot], dim=-1)

        # POST-PROCESSING
        elevation_optimized = best_elev_angle.detach().view(self.b, self.n_total, self.p, 1)
        
        if not self.optimize_poses:
            return self.init_poses_se3.tensor(), elevation_optimized
        
        best_delta_poses_se3 = pp.se3(best_delta_pose.detach()).Exp()
        base_poses = self.init_poses_se3[:, self.freeze_poses:, :]
        new_poses_se3 = base_poses @ best_delta_poses_se3
        frozen_poses = self.init_poses_se3[:, :self.freeze_poses, :]
        
        pose_optimized = torch.cat([frozen_poses.tensor(), new_poses_se3.tensor()], dim=1)

        return pose_optimized, elevation_optimized