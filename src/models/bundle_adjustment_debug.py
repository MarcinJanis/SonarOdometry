import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import transform_cart2polar, transform_polar2cart

import pypose as pp


class BundleAdjustment(nn.Module):
    def __init__(self, poses, patch_coords_r_theta, patch_coords_phi, sonar_param, freeze_poses=False):
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
            
        self.elevation_angle = nn.Parameter(patch_coords_phi) 

        # --- define constants parameters --- 
        self.patch_coords = patch_coords_r_theta
        self.sonar_param = sonar_param

    def transform(self, source_poses, target_poses, coords):
        source_poses = source_poses.squeeze(0)
        target_poses = target_poses.squeeze(0)
        coords = coords.squeeze(0)

        local_source_coords = transform_polar2cart(coords) 
        global_coords = source_poses @ local_source_coords
        local_target_coords = target_poses.Inv() @ global_coords
        coords = transform_cart2polar(local_target_coords)

        return coords.unsqueeze(0)
    
    def init_ba(self, source_poses_idx, target_poses_idx, patch_idx, delta, weights):
        
        # DEBUG CHECK: Zanim cokolwiek policzymy, sprawdźmy co dała sieć GRU
        if torch.isnan(delta).any(): print("[BA INIT DEBUG] Przekazana 'delta' z GRU zawiera NaN!")
        if torch.isnan(weights).any(): print("[BA INIT DEBUG] Przekazane 'weights' z GRU zawiera NaN!")

        self.source_poses_idx = source_poses_idx % self.pose_num
        self.target_poses_idx = target_poses_idx % self.pose_num
        self.patch_idx = patch_idx % self.edge_num

        with torch.no_grad():
            source_poses = self.poses[:, self.source_poses_idx, :].clone()
            target_poses = self.poses[:, self.target_poses_idx, :].clone()
            
            patch_coords = self.patch_coords[:, self.patch_idx, :]
            elevation_angle = self.elevation_angle[:, self.patch_idx].clone()
    
            source_coords = torch.cat([patch_coords, elevation_angle], dim=2)
            target_coords = self.transform(source_poses, target_poses, source_coords)
            
        self.target_coords = target_coords[:, :, :2] + self.scale_delta(delta)
        
        if torch.isnan(self.target_coords).any(): print("[BA INIT DEBUG] Wyliczone 'target_coords' zawiera NaN! (Prawdopodobnie wina delty lub transform_cart2polar)")

        self.init_poses = pp.SE3(self.poses.tensor().clone().detach())
        self.init_elevation_angle = self.elevation_angle.clone().detach()

        weights_param = weights.flatten()
        prior_weight = 1e-4

        weights_anchor_pose = torch.full((self.pose_num * 7,), prior_weight, device=weights.device, dtype=weights.dtype)
        weights_anchor_elev = torch.full((self.edge_num * 1,), prior_weight, device=weights.device, dtype=weights.dtype)
        
        weights = torch.cat([weights_param, weights_anchor_pose, weights_anchor_elev])
        self.weights = torch.diag(weights)

    # ====================================================================
    # PURE FORWARD - ZERO IFÓW, ZERO PRINTÓW (Bezpieczne dla vmap)
    # ====================================================================
    def forward(self, dummy_input=None):

        source_poses = self.poses[:, self.source_poses_idx, :]
        target_poses = self.poses[:, self.target_poses_idx, :]

        patch_coords = self.patch_coords[:, self.patch_idx % self.edge_num, :]
        elevation_angle = self.elevation_angle[:, self.patch_idx % self.edge_num, :]
        
        proj_coords = torch.cat([patch_coords, elevation_angle], dim=2)

        proj_coords = self.transform(source_poses, target_poses, proj_coords)

        residual_proj = proj_coords[:, :, :2] - self.target_coords
        residual_proj = self.scale_proj_err(residual_proj)
        residual_proj = residual_proj.view(1, -1)
        
        # Używamy SAFE OPTION - to wyklucza błąd logarytmu na kwaternionach dla vmap
        residual_pose = self.poses.tensor() - self.init_poses.tensor() 
        residual_pose = residual_pose.view(1, -1)
        
        residual_elev = self.elevation_angle - self.init_elevation_angle
        residual_elev = residual_elev.view(1, -1)
        
        residual = torch.cat([residual_proj, residual_pose, residual_elev], dim=1)

        return residual 

    # ====================================================================
    # DEBUG FORWARD - To wskaże nam winnego!
    # ====================================================================
    @torch.no_grad()
    def debug_forward(self, step):
        print(f"\n--- Uruchamiam Debug Forward (Krok optymalizacji: {step}) ---")
        source_poses = self.poses[:, self.source_poses_idx, :]
        target_poses = self.poses[:, self.target_poses_idx, :]
        
        patch_coords = self.patch_coords[:, self.patch_idx % self.edge_num, :]
        elevation_angle = self.elevation_angle[:, self.patch_idx % self.edge_num, :]
        
        proj_coords = torch.cat([patch_coords, elevation_angle], dim=2)
        if torch.isnan(proj_coords).any(): print("[DEBUG] NaN w proj_coords przed transformacją!")

        proj_coords = self.transform(source_poses, target_poses, proj_coords)
        if torch.isnan(proj_coords).any(): print("[DEBUG] NaN powstał WNĘTRZU funkcji self.transform()!")

        residual_proj = proj_coords[:, :, :2] - self.target_coords
        if torch.isnan(residual_proj).any(): print("[DEBUG] NaN w residual_proj (prawdopodobnie przez target_coords z delty)")
        
        residual_pose = self.poses.tensor() - self.init_poses.tensor() 
        if torch.isnan(residual_pose).any(): print("[DEBUG] NaN w residual_pose!")
        
        print("--- Koniec Debug Forward ---\n")


    def run(self, max_iter, early_stop_tol):
        
        strategy = pp.optim.strategy.TrustRegion(radius=1.0)
        optimizer = pp.optim.LM(self, strategy=strategy)

        prev_loss = float('inf')
        
        # Sprawdzamy początkowy stan przed jakimkolwiek krokiem LM
        self.debug_forward(step="INIT")

        for i in range(max_iter):
            try:
                # Włączamy gradient tylko do samego kroku (LM to wymaga)
                with torch.enable_grad():
                    loss = optimizer.step(input=None, weight=self.weights)

                if torch.isnan(loss):
                    print(f"[BA RUN] Loss stał się NaN w iteracji {i}!")
                    self.debug_forward(step=i) # Sprawdzamy co się zepsuło
                    break

                if abs(prev_loss - loss.item()) < early_stop_tol:
                    break
                    
                prev_loss = loss.item()

            except Exception as e:
                print(f"[BA CRASH] Optymalizator wyrzucił błąd w iteracji {i}:\n{e}")
                self.debug_forward(step=f"CRASH_AT_{i}")
                break
        
        return self.poses.tensor().view(self.b, self.n_act, 7), self.elevation_angle.view(self.b, self.n, self.p, 1)
    
    def scale_proj_err(self, proj_err):
        err_r = proj_err[:, :, 0] / (self.sonar_param.range.max - self.sonar_param.range.min) * self.sonar_param.resolution.bins
        err_t = proj_err[:, :, 1] / self.sonar_param.fov.horizontal * self.sonar_param.resolution.beams
        return torch.stack((err_r, err_t), dim=-1)

    def scale_delta(self, delta):
        delta_r = delta[:, 0] / self.sonar_param.resolution.bins * (self.sonar_param.range.max - self.sonar_param.range.min)
        delta_t = delta[:, 1] / self.sonar_param.resolution.beams * (self.sonar_param.fov.horizontal)
        return torch.stack((delta_r, delta_t), dim=-1)