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
                 damping=True,
                 prior_scale_trans=50000.0,  # Nowy parametr
                 prior_scale_rot=10000.0):   # Nowy parametr
        
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
        
        # --- ZARZĄDZANIE GRAFEM ---
        self.E = self.s_idx.shape[0]
        self.M = self.b * self.n_total * self.p  
        self.patch_idx = (patch_idx % self.M).squeeze()
        
        init_poses = init_poses.clone()
        init_poses[:, :, 3:] = torch.nn.functional.normalize(init_poses[:, :, 3:], p=2, dim=-1)
        self.init_poses_se3 = pp.SE3(init_poses.view(poses_n, 7))
        self.current_poses_se3 = self.init_poses_se3.clone()
        
        self.current_elev_phi = init_patch_coords_phi.view(self.M).clone()

        flat_r_theta = init_patch_coords_r_theta.view(1, -1, 2)[:, self.patch_idx, :].squeeze()
        self.r_obs = flat_r_theta[:, 0]
        self.theta_obs = flat_r_theta[:, 1]
        
        self.weights = weights.view(-1, 2) 

        self.num_optim = self.act_n - self.freeze_poses
        self.optimize_poses = self.num_optim > 0

        # --- Inicjalizacja Usztywnionego Prioru ---
        self.damping = damping
        self.prior_w_trans = (2.63 / 50) * prior_scale_trans
        self.prior_w_rot = (356.51 / 50) * prior_scale_rot

        with torch.no_grad():
            self.coords_baseline = self._compute_baseline(delta.view(-1, 2))

    # def _compute_baseline(self, delta_fls):
    #     """
    #     Wylicza obiektywny punkt docelowy na podstawie obserwacji sonaru.
    #     Cel jest w 100% niezależny od zaszumionych estymat początkowych póz.
    #     """
    #     # 1. Bierzemy oryginalne współrzędne (r, theta) punktu obserwowanego w klatce źródłowej.
    #     #    To są twarde pomiary z klatki 1.
    #     source_physical = torch.stack([self.r_obs, self.theta_obs], dim=1)
        
    #     # 2. Skalujemy je do pikseli obrazu sonaru (tak jak w delcie)
    #     source_fls = self.scale_phisical2fls(source_physical)
        
    #     # 3. Dodajemy wektor przepływu (flow).
    #     #    To daje nam fizyczny piksel w klatce docelowej, który namierzyła sieć.
    #     #    To jest nasz betonowy "obserwowany punkt docelowy".
    #     target_fls = source_fls + delta_fls 
        
    #     return target_fls
    
    def _compute_baseline(self, delta_fls):
        phi_edge = self.current_elev_phi[self.patch_idx]
        
        x_s = self.r_obs * torch.cos(phi_edge) * torch.cos(self.theta_obs)
        y_s = self.r_obs * torch.cos(phi_edge) * torch.sin(self.theta_obs)
        z_s = self.r_obs * torch.sin(phi_edge)
        P_s = torch.stack([x_s, y_s, z_s], dim=1) 

        T_s_init = pp.SE3(self.current_poses_se3.tensor()[self.s_idx])
        T_t_init = pp.SE3(self.current_poses_se3.tensor()[self.t_idx])
        P_t = (T_t_init.Inv() @ T_s_init) @ P_s 
        
        x_t, y_t, z_t = P_t[:, 0], P_t[:, 1], P_t[:, 2]
        
        r_hat = torch.sqrt(x_t**2 + y_t**2 + z_t**2 + 1e-8)
        theta_hat = torch.atan2(y_t, x_t)
        
        p_hat_fls = self.scale_phisical2fls(torch.stack([r_hat, theta_hat], dim=1))
        return p_hat_fls + delta_fls

    def compute_huber_cost(self, e_net, delta_huber):
        e_norm = torch.norm(e_net, dim=1)
        mask = e_norm < delta_huber
        cost = torch.where(mask, 0.5 * e_norm**2, delta_huber * (e_norm - 0.5 * delta_huber))
        return torch.sum(cost).item()

    def compute_residuals_and_jacobians(self, delta_huber=1.5, r_penalty_factor=50.0):
        phi_edge = self.current_elev_phi[self.patch_idx]
        
        x_s = self.r_obs * torch.cos(phi_edge) * torch.cos(self.theta_obs)
        y_s = self.r_obs * torch.cos(phi_edge) * torch.sin(self.theta_obs)
        z_s = self.r_obs * torch.sin(phi_edge)
        P_s = torch.stack([x_s, y_s, z_s], dim=1)

        T_s = pp.SE3(self.current_poses_se3.tensor()[self.s_idx])
        T_t = pp.SE3(self.current_poses_se3.tensor()[self.t_idx])
        T_ts = T_t.Inv() @ T_s
        R_ts = T_ts.rotation().matrix()

        P_t = T_ts @ P_s
        x_t, y_t, z_t = P_t[:, 0], P_t[:, 1], P_t[:, 2]

        eps = 1e-8
        r_sq_2d = x_t**2 + y_t**2 + eps
        r_hat = torch.sqrt(r_sq_2d + z_t**2)
        theta_hat = torch.atan2(y_t, x_t)
        
        p_hat_fls = self.scale_phisical2fls(torch.stack([r_hat, theta_hat], dim=1))
        
        # --- Błąd Surowy i Balans (Anizotropia) ---
        e_raw = self.coords_baseline - p_hat_fls 
        
        # Zewnętrzny parametr kary za zasięg
        balance_weights = torch.tensor([r_penalty_factor, 1.0], device=self.device).view(1, 2)
        
        e_net = e_raw * self.weights * balance_weights 
        
        e_norm = torch.norm(e_net, dim=1, keepdim=True) 
        w_robust = torch.where(e_norm < delta_huber, torch.ones_like(e_norm), delta_huber / (e_norm + 1e-6))
        
        W_total = self.weights * w_robust 
        W_sqrt = torch.sqrt(W_total)
        
        e = (e_raw * balance_weights * W_sqrt).unsqueeze(-1) 

        # --- Jakobiany ---
        S_r = self.fls_h / (self.r_max - self.r_min)
        S_theta = self.fls_w / self.fov_horizontal
        
        minus_J_pi_phys = -1.0 * torch.stack([
            torch.stack([x_t/r_hat, y_t/r_hat, z_t/r_hat], dim=1),
            torch.stack([-y_t/r_sq_2d, x_t/r_sq_2d, torch.zeros_like(x_t)], dim=1)
        ], dim=1)
        
        Scale_mat = torch.tensor([[S_r * r_penalty_factor, 0], [0, S_theta]], device=self.device)
        minus_J_pi = Scale_mat @ minus_J_pi_phys 
        minus_J_pi = W_sqrt.unsqueeze(-1) * minus_J_pi 

        I_3 = torch.eye(3, device=self.device).unsqueeze(0).expand(self.E, -1, -1)
        J_p = torch.zeros(self.E, 2, 6 * self.num_optim, device=self.device)
        
        if self.optimize_poses:
            def get_skew(v):
                x_v, y_v, z_v = v[:, 0], v[:, 1], v[:, 2]
                zeros = torch.zeros_like(x_v)
                return torch.stack([
                    torch.stack([zeros, -z_v, y_v], dim=1),
                    torch.stack([z_v, zeros, -x_v], dim=1),
                    torch.stack([-y_v, x_v, zeros], dim=1)
                ], dim=1)

            skew_Pt = get_skew(P_t)
            J_Tt = minus_J_pi @ torch.cat([-I_3, skew_Pt], dim=2)
            
            skew_Ps = get_skew(P_s)
            J_Ts = minus_J_pi @ R_ts @ torch.cat([I_3, -skew_Ps], dim=2)

            s_opt_idx = self.s_idx - self.freeze_poses
            t_opt_idx = self.t_idx - self.freeze_poses

            for k in range(self.num_optim):
                mask_s = (s_opt_idx == k)
                if mask_s.any(): J_p[mask_s, :, k*6 : k*6+6] = J_Ts[mask_s]
                
                mask_t = (t_opt_idx == k)
                if mask_t.any(): J_p[mask_t, :, k*6 : k*6+6] = J_Tt[mask_t]

        dp_dphi = torch.stack([
            -self.r_obs * torch.sin(phi_edge) * torch.cos(self.theta_obs),
            -self.r_obs * torch.sin(phi_edge) * torch.sin(self.theta_obs),
             self.r_obs * torch.cos(phi_edge)
        ], dim=1).unsqueeze(-1)
        
        J_phi = minus_J_pi @ R_ts @ dp_dphi 

        return e, J_p, J_phi, e_net

    def compute_prior(self):
        if not self.damping or not self.optimize_poses:
            return 0.0, 0, 0
            
        T_init = pp.SE3(self.init_poses_se3.tensor()[self.freeze_poses:])
        T_curr = pp.SE3(self.current_poses_se3.tensor()[self.freeze_poses:])
        
        e_prior_se3 = (T_init.Inv() @ T_curr).Log() 
        
        W_diag = torch.tensor([self.prior_w_trans]*3 + [self.prior_w_rot]*3, device=self.device)
        cost_prior = torch.sum(W_diag * e_prior_se3**2).item()
        
        g_prior = -1.0 * W_diag * e_prior_se3 
        H_prior = torch.diag(W_diag.repeat(self.num_optim))
        
        return cost_prior, H_prior, g_prior.view(-1)

    def run(self, max_iter=10, initial_lambda=1e-3, delta_huber=2.0, 
            r_penalty_factor=50.0, phi_clamp_min=-1.2, phi_clamp_max=1.2, 
            disp_stats=False):
        """Uruchamia optymalizację z wyeksponowanymi parametrami sterującymi."""
        lambda_lm = initial_lambda
        current_loss = float('inf')
        dim_p = 6 * self.num_optim
        
        for i in range(max_iter):
            with torch.no_grad():
                # Przekazujemy r_penalty_factor do obliczeń
                e, J_p, J_l, e_net = self.compute_residuals_and_jacobians(delta_huber, r_penalty_factor)
                
                cost_obs = self.compute_huber_cost(e_net, delta_huber)
                cost_prior, H_prior, g_prior_vec = self.compute_prior()
                current_loss = cost_obs + cost_prior
                
                if disp_stats:
                    r_err = e[:, 0, 0].abs().mean().item()
                    th_err = e[:, 1, 0].abs().mean().item()
                    print(f'Iter {i:02d} | Loss: {current_loss:.4f} | r_e: {r_err:.4f} | th_e: {th_err:.4f} | lam: {lambda_lm:.1e}')

                # ====== SCHUR COMPLEMENT ======
                H_ll_edge = J_l.transpose(1, 2) @ J_l      
                H_pl_edge = J_p.transpose(1, 2) @ J_l      
                
                g_l_edge  = -J_l.transpose(1, 2) @ e        
                g_p_edge  = -J_p.transpose(1, 2) @ e        
                
                idx_11 = self.patch_idx.view(self.E, 1, 1)
                
                H_ll_patch = torch.zeros(self.M, 1, 1, device=self.device)
                H_ll_patch.scatter_add_(0, idx_11, H_ll_edge)
                
                g_l_patch = torch.zeros(self.M, 1, 1, device=self.device)
                g_l_patch.scatter_add_(0, idx_11, g_l_edge)
                
                idx_pl = self.patch_idx.view(self.E, 1, 1).expand(self.E, dim_p, 1)
                H_pl_patch = torch.zeros(self.M, dim_p, 1, device=self.device)
                H_pl_patch.scatter_add_(0, idx_pl, H_pl_edge)

                H_ll_inv_patch = 1.0 / (H_ll_patch + lambda_lm) 
                
                if self.optimize_poses:
                    H_pp_edge = J_p.transpose(1, 2) @ J_p 
                    H_global = torch.sum(H_pp_edge, dim=0) 
                    g_global = torch.sum(g_p_edge, dim=0)  
                    
                    H_schur = H_pl_patch @ H_ll_inv_patch @ H_pl_patch.transpose(1, 2) 
                    g_schur = H_pl_patch @ H_ll_inv_patch @ g_l_patch 
                    
                    H_global -= torch.sum(H_schur, dim=0)
                    g_global -= torch.sum(g_schur, dim=0)
                    
                    if self.damping:
                        H_global += H_prior
                        g_global += g_prior_vec.unsqueeze(-1)
                        
                    H_global_lm = H_global + lambda_lm * torch.diag(torch.diag(H_global))
                    
                    dx_p = torch.linalg.solve(H_global_lm, g_global) 
                    dx_p_expanded = dx_p.unsqueeze(0).expand(self.M, dim_p, 1)
                    
                    dx_l_patch = H_ll_inv_patch @ (g_l_patch - H_pl_patch.transpose(1, 2) @ dx_p_expanded)
                else:
                    dx_p = None
                    dx_l_patch = H_ll_inv_patch @ g_l_patch

                old_poses = self.current_poses_se3.clone()
                old_phi = self.current_elev_phi.clone()

                # AKTUALIZACJA I ZABEZPIECZENIE (Clamping elewacji)
                new_phi = self.current_elev_phi + dx_l_patch.squeeze(-1).squeeze(-1)
                
                # Zewnętrzne parametry dla clamingu
                self.current_elev_phi = torch.clamp(new_phi, min=phi_clamp_min, max=phi_clamp_max)
                
                if self.optimize_poses:
                    dx_p_reshaped = dx_p.view(self.num_optim, 6)
                    
                    opt_poses = pp.SE3(self.current_poses_se3.tensor()[self.freeze_poses:])
                    frozen_poses = pp.SE3(self.current_poses_se3.tensor()[:self.freeze_poses])
                    
                    update_se3 = pp.se3(dx_p_reshaped).Exp()
                    new_opt_poses = opt_poses @ update_se3 
                    
                    self.current_poses_se3 = pp.SE3(torch.cat([frozen_poses.tensor(), new_opt_poses.tensor()], dim=0))
                    
                # Trust Region Check
                _, _, _, new_e_net = self.compute_residuals_and_jacobians(delta_huber, r_penalty_factor)
                new_cost_obs = self.compute_huber_cost(new_e_net, delta_huber)
                new_cost_prior, _, _ = self.compute_prior()
                new_loss = new_cost_obs + new_cost_prior
                
                if new_loss < current_loss:
                    lambda_lm = max(1e-7, lambda_lm / 5.0)
                    current_loss = new_loss
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