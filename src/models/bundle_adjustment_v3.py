import torch
import torch.nn as nn
import pypose as pp

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
        
        self.s_idx = (source_frame_idx % poses_n).squeeze()
        self.t_idx = (target_frame_idx % poses_n).squeeze()
        self.patch_idx = (patch_idx % (self.b * self.act_n * self.p)).squeeze()
        
        init_poses = init_poses.clone()
        init_poses[:, :, 3:] = torch.nn.functional.normalize(init_poses[:, :, 3:], p=2, dim=-1)
        self.current_poses_se3 = pp.SE3(init_poses.view(poses_n, 7))
        
        flat_phi = init_patch_coords_phi.view(1, -1, 1)[:, self.patch_idx, :].squeeze()
        self.current_elev_phi = flat_phi.clone()

        self.E = self.patch_idx.shape[0]
        flat_r_theta = init_patch_coords_r_theta.view(1, -1, 2)[:, self.patch_idx, :].squeeze()
        self.r_obs = flat_r_theta[:, 0]
        self.theta_obs = flat_r_theta[:, 1]
        
        # POPRAWKA 1: Wagi i Delta są już w kształcie (E, 2)
        # Indeks 0 -> Waga/Delta dla 'r'
        # Indeks 1 -> Waga/Delta dla 'theta'
        self.weights = weights.view(-1, 2) 

        self.num_optim = self.act_n - self.freeze_poses
        self.optimize_poses = self.num_optim > 0

        # POPRAWKA 2: Przekazujemy Deltę o kształcie (E, 2) bezpośrednio do bazy
        with torch.no_grad():
            self.coords_baseline = self._compute_baseline(delta.view(-1, 2))


    def _compute_baseline(self, delta_fls):
        """Wylicza e_ref = reproject(P_init) + delta_GRU"""
        x_s = self.r_obs * torch.cos(self.current_elev_phi) * torch.cos(self.theta_obs)
        y_s = self.r_obs * torch.cos(self.current_elev_phi) * torch.sin(self.theta_obs)
        z_s = self.r_obs * torch.sin(self.current_elev_phi)
        P_s = torch.stack([x_s, y_s, z_s], dim=1).unsqueeze(-1) # (E, 3, 1)

        T_s_init = self.current_poses_se3[self.s_idx]
        T_t_init = self.current_poses_se3[self.t_idx]
        P_t = (T_t_init.Inv() @ T_s_init) @ P_s
        
        x_t, y_t, z_t = P_t[:, 0, 0], P_t[:, 1, 0], P_t[:, 2, 0]
        r_hat = torch.sqrt(x_t**2 + y_t**2 + z_t**2)
        theta_hat = torch.atan2(y_t, x_t)
        
        p_hat_phys = torch.stack([r_hat, theta_hat], dim=1)
        p_hat_fls = self.scale_phisical2fls(p_hat_phys)
        
        # Dodajemy deltę. Obydwa mają (E, 2), więc 'r' dodaje się do 'r', a 'theta' do 'theta'
        return p_hat_fls + delta_fls 


    def compute_residuals_and_jacobians(self):
        """Zbatczowane wyliczenie błędów i wszystkich Jakobianów analitycznych."""
        x_s = self.r_obs * torch.cos(self.current_elev_phi) * torch.cos(self.theta_obs)
        y_s = self.r_obs * torch.cos(self.current_elev_phi) * torch.sin(self.theta_obs)
        z_s = self.r_obs * torch.sin(self.current_elev_phi)
        P_s = torch.stack([x_s, y_s, z_s], dim=1).unsqueeze(-1)

        T_s = self.current_poses_se3[self.s_idx]
        T_t = self.current_poses_se3[self.t_idx]
        T_ts = T_t.Inv() @ T_s
        R_ts = T_ts.rotation().matrix()

        P_t = T_ts @ P_s
        x_t, y_t, z_t = P_t[:, 0, 0], P_t[:, 1, 0], P_t[:, 2, 0]

        r_hat = torch.sqrt(x_t**2 + y_t**2 + z_t**2)
        theta_hat = torch.atan2(y_t, x_t)
        p_hat_fls = self.scale_phisical2fls(torch.stack([r_hat, theta_hat], dim=1))
        
        # --- POPRAWKA 3: Implementacja Wag (E, 2) ---
        # Błąd surowy (E, 2)
        e_raw = self.coords_baseline - p_hat_fls 
        
        # Mnożenie element-wise: r_err * waga_r, theta_err * waga_theta (Wymiar dalej E, 2)
        e = e_raw * self.weights 
        e = e.unsqueeze(-1) # (E, 2, 1) gotowe do mnożeń macierzowych!
        
        S_r = self.fls_h / (self.r_max - self.r_min)
        S_theta = self.fls_w / self.fov_horizontal
        
        minus_J_pi_phys = -1.0 * torch.stack([
            torch.stack([x_t/r_hat, y_t/r_hat, z_t/r_hat], dim=1),
            torch.stack([-y_t/(x_t**2 + y_t**2), x_t/(x_t**2 + y_t**2), torch.zeros_like(x_t)], dim=1)
        ], dim=1)
        
        Scale_mat = torch.tensor([[S_r, 0], [0, S_theta]], device=self.device)
        minus_J_pi = Scale_mat @ minus_J_pi_phys # (E, 2, 3)
        
        # Ważenie Jakobianu Projekcji!
        # self.weights ma (E, 2). unsqueeze(-1) daje (E, 2, 1).
        # Gdy mnożymy (E, 2, 1) * (E, 2, 3), PyTorch inteligentnie mnoży cały PIERWSZY wiersz Jakobianu 
        # (odpowiadający za r) przez wagę r, a cały DRUGI wiersz (theta) przez wagę theta.
        minus_J_pi = self.weights.unsqueeze(-1) * minus_J_pi 

        I_3 = torch.eye(3, device=self.device).unsqueeze(0).expand(self.E, -1, -1)

        J_p = torch.zeros(self.E, 2, 6 * self.num_optim, device=self.device)
        
        if self.optimize_poses:
            skew_Pt = pp.skew(P_t.squeeze(-1))
            J_Tt = minus_J_pi @ torch.cat([-I_3, skew_Pt], dim=2)
            
            skew_Ps = pp.skew(P_s.squeeze(-1))
            J_Ts = minus_J_pi @ R_ts @ torch.cat([I_3, -skew_Ps], dim=2)

            s_opt_idx = self.s_idx - self.freeze_poses
            t_opt_idx = self.t_idx - self.freeze_poses

            for k in range(self.num_optim):
                mask_s = (s_opt_idx == k)
                if mask_s.any(): J_p[mask_s, :, k*6 : k*6+6] = J_Ts[mask_s]
                
                mask_t = (t_opt_idx == k)
                if mask_t.any(): J_p[mask_t, :, k*6 : k*6+6] = J_Tt[mask_t]

        dp_dphi = torch.stack([
            -self.r_obs * torch.sin(self.current_elev_phi) * torch.cos(self.theta_obs),
            -self.r_obs * torch.sin(self.current_elev_phi) * torch.sin(self.theta_obs),
             self.r_obs * torch.cos(self.current_elev_phi)
        ], dim=1).unsqueeze(-1)
        
        J_phi = minus_J_pi @ R_ts @ dp_dphi # (E, 2, 1)

        return e, J_p, J_phi


    def run(self, max_iter=10, min_delta=1e-4, initial_lambda=1e-3, disp_stats=False):
        """Własny optymalizator Levenberga-Marquardta ze Schur Complement"""
        lambda_lm = initial_lambda
        best_loss = float('inf')
        
        for i in range(max_iter):
            with torch.no_grad():
                e, J_p, J_l = self.compute_residuals_and_jacobians()
                
                current_loss = torch.sum(e**2).item()
                
                if disp_stats:
                    r_err = e[:, 0, 0].abs().mean().item()
                    th_err = e[:, 1, 0].abs().mean().item()
                    print(f'Iter {i:02d} | LM-Loss: {current_loss:.4f} | r_err: {r_err:.4f} | th_err: {th_err:.4f} | lambda: {lambda_lm:.1e}')

                if current_loss + min_delta < best_loss and i > 0:
                    pass 
                elif current_loss < best_loss:
                    best_loss = current_loss
                
                H_ll = J_l.transpose(1, 2) @ J_l # (E, 1, 1)
                # ====== This:
                H_ll_inv = 1.0 / (H_ll + lambda_lm) # (E, 1, 1)
                # or this: 
                # H_ll_inv = 1.0 / (H_ll + lambda_lm * H_ll)

                
                H_pl = J_p.transpose(1, 2) @ J_l # (E, 36, 1)
                H_pp = J_p.transpose(1, 2) @ J_p # (E, 36, 36)
                
                g_p_edge = J_p.transpose(1, 2) @ e # (E, 36, 1)
                g_l_edge = J_l.transpose(1, 2) @ e # (E, 1, 1)
                
                if self.optimize_poses:
                    H_p_marg = H_pp - H_pl @ H_ll_inv @ H_pl.transpose(1, 2) 
                    g_p_marg = g_p_edge - H_pl @ H_ll_inv @ g_l_edge         
                    
                    H_global = torch.sum(H_p_marg, dim=0) 
                    g_global = torch.sum(g_p_marg, dim=0) 
                    
                    H_global_lm = H_global + lambda_lm * torch.diag(torch.diag(H_global))
                    
                    dx_p = torch.linalg.solve(H_global_lm, g_global) 
                    
                    dx_p_batched = dx_p.unsqueeze(0).expand(self.E, -1, -1)
                    dx_l = H_ll_inv @ (g_l_edge - H_pl.transpose(1, 2) @ dx_p_batched) 
                else:
                    dx_p = None
                    dx_l = H_ll_inv @ g_l_edge

                old_poses = self.current_poses_se3.clone()
                old_phi = self.current_elev_phi.clone()

                self.current_elev_phi = self.current_elev_phi + dx_l.squeeze()
                
                if self.optimize_poses:
                    dx_p_reshaped = dx_p.view(self.num_optim, 6)
                    opt_poses = self.current_poses_se3[self.freeze_poses:]
                    new_opt_poses = opt_poses.Retr(dx_p_reshaped) 
                    
                    frozen_poses = self.current_poses_se3[:self.freeze_poses]
                    self.current_poses_se3 = pp.SE3(torch.cat([frozen_poses.tensor(), new_opt_poses.tensor()], dim=0))

                new_e, _, _ = self.compute_residuals_and_jacobians()
                new_loss = torch.sum(new_e**2).item()
                
                if new_loss < current_loss:
                    lambda_lm = max(1e-7, lambda_lm / 10.0)
                else:
                    lambda_lm = min(1e4, lambda_lm * 10.0)
                    self.current_poses_se3 = old_poses
                    self.current_elev_phi = old_phi

        pose_optimized = self.current_poses_se3.tensor().view(self.b, self.act_n, 7)
        elevation_optimized = self.current_elev_phi.view(self.b, self.n_total, self.p, 1)

        return pose_optimized, elevation_optimized


    def scale_phisical2fls(self, coords):
        r_norm = (coords[:, 0] - self.r_min) / (self.r_max - self.r_min)
        r = r_norm * self.fls_h
        theta_norm = coords[:, 1] / self.fov_horizontal 
        theta = (theta_norm + 0.5) * self.fls_w
        return torch.stack([r, theta], dim=1)


