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
        self.freeze_poses = freeze_poses
        
        # physical to fls units scaling 
        self.physic2fls_scale_factor = torch.tensor(
            [sonar_param.resolution.bins / (sonar_param.range.max - sonar_param.range.min),
             sonar_param.resolution.beams / sonar_param.fov.horizontal], 
             device = self.device
        ).view(1, 1, 2)

        # remember input shape:
        self.b, self.n_total, self.p, _ = init_patch_coords_r_theta.shape
        self.act_n = init_poses.shape[1]
        poses_n = self.b * self.act_n
        self.edges_n = self.b * self.act_n * self.p
        self.edges_total = self.b * self.n_total * self.p

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
        elif freeze_poses == 0:
            self.poses_anchor = pp.Parameter(init_poses_se3) 
            self.split_poses = False
        else:
            self.poses_anchor = init_poses_se3[:, :freeze_poses, :]
            self.poses_optim = pp.Parameter(init_poses_se3[:, freeze_poses:, :])
            self.split_poses = True
            
        # --- GLOBAL IDX -> LOCAL IDX ---
        self.source_frame_idx = source_frame_idx % poses_n
        self.target_frame_idx = target_frame_idx % poses_n
        self.patch_idx = patch_idx % self.edges_n

        # =========================================================================
        # RZADKIE OPTYMALIZOWANIE (SPARSE / ACTIVE WINDOW TRICK)
        # =========================================================================
        # Wyciągamy indeksy wyłącznie tych punktów, które aktualnie tworzą krawędzie.
        self.unique_patch_idx, self.inverse_patch_idx = torch.unique(self.patch_idx, return_inverse=True)
        
        # Tworzymy Parametr tylko z WYCINKA tensora. 
        # Rozmiar macierzy Jakobianu spada drastycznie.
        active_elevations = patch_coords_phi[:, self.unique_patch_idx, :].clone()
        self.elevation_angle_active = nn.Parameter(active_elevations)
        
        # Zapisujemy pełną sekwencję (nie-parametr) by oddać ją na koniec po aktualizacji
        self.elevation_angle_full = patch_coords_phi.clone().detach()
        # =========================================================================

        # --- define parameters not optimized --- 
        self.patch_coords_r_theta = patch_coords_r_theta
        self.sonar_param = sonar_param

        # --- projection base line ---
        source_poses = init_poses_se3[:, self.source_frame_idx, :].clone() 
        target_poses = init_poses_se3[:, self.target_frame_idx, :].clone() 
        
        patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :] 
        
        # Pobieramy elewację do wyliczenia baseline (jeszcze nie zoptymalizowaną)
        elevation_angle_base = active_elevations[:, self.inverse_patch_idx, :].clone() 
        source_coords = torch.cat([patch_coords, elevation_angle_base], dim = 2)
    
        target_coords = transform(source_poses, target_poses, source_coords)
            
        self.coords_baseline = (target_coords[:, :, :2] * self.physic2fls_scale_factor + delta).detach()
        self.weights_1d = weights.flatten().detach()

    def forward(self, dummy_input=None):

        # compose pose tensor and coord tensor
        if self.split_poses:
            poses = torch.cat([self.poses_anchor, self.poses_optim], dim=1)
        else:
            poses = self.poses_anchor

        patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :]
        
        # Pobieramy aktywne, optymalizowane kąty, rekonstruując ich miejsca w grafie
        elevation_angle = self.elevation_angle_active[:, self.inverse_patch_idx, :] 
        source_coords = torch.cat([patch_coords, elevation_angle], dim = 2)

        # expand for all edges
        source_poses = poses[:, self.source_frame_idx, :]
        target_poses = poses[:, self.target_frame_idx, :]

        # --- project --- 
        projected_coords = transform(source_poses, target_poses, source_coords)

        # --- projection error ---
        project_err = (projected_coords[:, :, :2] * self.physic2fls_scale_factor - self.coords_baseline) 
        project_err = project_err.view(1, -1)
        
        weighted_err = project_err * torch.sqrt(self.weights_1d + 1e-8)
       
        return weighted_err

    def run(self, max_iter=2, early_stop_tol=1e-4, trust_region=2.0):
        
        # Zbieramy tylko aktywne parametry do optymalizacji
        params_to_opt = [self.elevation_angle_active]
        if self.split_poses:
            params_to_opt.append(self.poses_optim)
        elif self.freeze_poses == 0:
            params_to_opt.append(self.poses_anchor)

        # Używamy lekkiego i błyskawicznego Adama zamiast ciężkiego LM
        # LR (learning rate) ustalamy na relatywnie wysoki (np. 0.05), bo mamy tylko 2 iteracje, 
        # aby dogonić wynik docelowy (deltę z GRU). W razie potrzeby możesz go zmniejszyć.
        optimizer = torch.optim.Adam(params_to_opt, lr=0.05)

        with torch.enable_grad(): 
            for i in range(max_iter):   
                optimizer.zero_grad()
                err = self.forward()
                
                # Błąd to po prostu suma kwadratów reszt (Residual Sum of Squares)
                loss = (err ** 2).sum()
                
                # Błyskawiczny, natywny backward PyTorcha (bez macierzy Jakobianu!)
                loss.backward()
                optimizer.step()
        
        # --- Zbieranie zaktualizowanych danych ---
        if self.split_poses:
            pose_optimized_se3 = torch.cat([self.poses_anchor, self.poses_optim], dim=1)
        else:
            pose_optimized_se3 = self.poses_anchor

        pose_optimized = pose_optimized_se3.tensor().detach().view(self.b, self.act_n, 7)
        
        # Bezpieczna aktualizacja tylko tych kątów, które były w oknie
        with torch.no_grad():
            self.elevation_angle_full[:, self.unique_patch_idx, :] = self.elevation_angle_active.detach()
            
        elevation_optimized = self.elevation_angle_full.view(self.b, self.n_total, self.p, 1)

        return pose_optimized, elevation_optimized

def transform(source_poses, target_poses, coords):
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