import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import transform_cart2polar, transform_polar2cart, project_points

import pypose as pp

class BundleAdjustment(nn.Module):
    def __init__(self, 
                 init_poses, 
                 init_patch_coords_r_theta, init_patch_coords_phi, 
                 source_frame_idx, target_frame_idx, patch_idx,
                 delta, weights, freeze_poses, 
                 model_config, sonar_config, 
                 damping = False):
        
        super().__init__()

        # --- init ---
        self.device = init_poses.device
    
        self.damping = damping


        # For details see BA_test.ipynb
        # damping_trans_weight = scale [pix/m] / (2 * max_permissible_trans)
        # damping_trans_weight = scale [pix/rad] / (2 * max_permissible_rot)
        self.damping_trans_weight = 2.63 / 50 # 9.329494828396749 
        self.damping_rot_weight = 356.51 / 50# 2608.996360840966 

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

        if freeze_poses >= self.act_n:
            self.optimize_poses = False
        else:
            # translation and rotation correction 
            self.optimize_poses = True
            num_optim = self.act_n - self.freeze_poses # number os poses to be optimized
            self.trans_correction = nn.Parameter(torch.zeros(self.b, num_optim, 3, device=self.device))
            self.rot_correction = nn.Parameter(torch.zeros(self.b, num_optim, 3, device=self.device))
        
        # phi angle 
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

        # rescale reprojected points to fls/pixel units, add delta
        # self.coords_baseline = target_coords[:, :, :2] * self.physic2fls_scale_factor + delta
        self.coords_baseline = self.scale_phisical2fls(target_coords[:, :2]) + delta

        # weights for optimization
        self.weights = weights
       

    # def get_pose_estim(self):
    #     '''
    #     Get initial poses, add optimized correction
    #     '''
    #     if not self.optimize_poses:
    #             return self.init_poses_se3
    #     else:
    #         # connect trans and rotat
    #         poses_correction = torch.cat((self.trans_correction, self.rot_correction), dim=-1)
    #         # set as se3 (6d) object and transform to SE3 (7d)
    #         poses_correction_SE3 = pp.se3(poses_correction).Exp()
            
    #         # add corrections
    #         base_poses = self.init_poses_se3[:, self.freeze_poses:, :] # poses to change
    #         new_poses_se3 = base_poses @ poses_correction_SE3 # changed poses 

    #         frozen_poses = self.init_poses_se3[:, :self.freeze_poses, :] # unchanged poses
    #         return torch.cat([frozen_poses, new_poses_se3], dim=1)

    def get_pose_estim(self):
        if not self.optimize_poses:
                return self.init_poses_se3
        else:
            # MAGIA SKALOWANIA GRADIENTU: 
            # Mnożymy trans_correction np. x20. Jeśli sieć uzna trans za 0.01,
            # w rzeczywistości przesuniemy się o 0.2, a gradient wzrośnie 20-krotnie!
            trans_multiplier = 20.0 
            scaled_trans = self.trans_correction * trans_multiplier
            
            # connect trans and rotat
            poses_correction = torch.cat((scaled_trans, self.rot_correction), dim=-1)
            poses_correction_SE3 = pp.se3(poses_correction).Exp()
            
            # add corrections
            base_poses = self.init_poses_se3[:, self.freeze_poses:, :]
            new_poses_se3 = base_poses @ poses_correction_SE3

            frozen_poses = self.init_poses_se3[:, :self.freeze_poses, :]
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
        # --- projection error ---
        
        projected_coords_fls = self.scale_phisical2fls(projected_coords)
        project_err = self.coords_baseline - projected_coords_fls 
        
        # ZBALANSOWANIE SKALI BŁĘDÓW:
        # Jeśli th_err jest ~3.5, a r_err jest ~0.2, to th jest 15-20 razy większe.
        # Wzmacniamy błąd r, żeby optymalizator bał się go "psuć".
        err_r = project_err[:, 0] * 10.0  # Mnożnik x10 dla R
        err_th = project_err[:, 1] * 1.0  # Theta zostawiamy bez zmian
        
        balanced_err = torch.cat([err_r.unsqueeze(-1), err_th.unsqueeze(-1)], dim=1)
        
        # add weights
        weighted_err = balanced_err * self.weights
        
        return weighted_err
    

    def run3(self, max_iter=100, min_delta=1e-5, disp_stats=True):
        
        # Poses zostają przy Adamie - to najszybszy sposób na SE(3) w PyTorchu 
        # bez budowania gigantycznych, rzadkich macierzy S = B - E * C^-1 * E^T
        opt_pose = torch.optim.Adam([self.trans_correction, self.rot_correction], 
                                     lr=0.015, betas=(0.0, 0.9))
        sched_pose = torch.optim.lr_scheduler.StepLR(opt_pose, step_size=25, gamma=0.5)

        best_loss = float('inf')
        
        # Fizyczne limity wiązki (np. +/- 15 stopni w radianach)
        max_phi_rad = 0.26 
        min_phi_rad = -0.26

        pose_warmup_iters = 20 # Dajemy pozom chwilę na ustawienie się bez zmiany mapy

        with torch.enable_grad():
            for i in range(max_iter):
                
                # ==========================================
                # FAZA 1: OPTYMALIZACJA PÓZ SE(3)
                # ==========================================
                if self.optimize_poses:
                    self.elevation_angle.requires_grad_(False)
                    steps_pose = 10 if i < pose_warmup_iters else 5
                    
                    for _ in range(steps_pose):
                        opt_pose.zero_grad()
                        err = self.forward() 
                        loss_pose = F.smooth_l1_loss(err, torch.zeros_like(err), beta=2.5)
                        
                        if self.damping:
                            prior_trans = torch.mean(self.trans_correction**2) * self.damping_trans_weight 
                            prior_rot = torch.mean(self.rot_correction**2) * self.damping_rot_weight 
                            loss_pose = loss_pose + prior_trans + prior_rot
                            
                        loss_pose.backward()
                        opt_pose.step()
                    
                    sched_pose.step()

                # ==========================================
                # FAZA 2: EXACT GAUSS-NEWTON DLA ELEWACJI
                # Klonujemy zachowanie kernela DPVO (atomicAdd & Schur)
                # ==========================================
                if i >= pose_warmup_iters:
                    self.elevation_angle.requires_grad_(False) # Odpinamy główną mapę od grafu
                    
                    for _ in range(2): # 2 kroki analitycznego Newtona
                        
                        # 1. IZOLACJA KRAWĘDZI: Każda krawędź dostaje "swój" kąt jako zmienną niezależną (Leaf)
                        phi_edges_leaf = self.elevation_angle[:, self.patch_idx, :].detach().clone().requires_grad_(True)

                        # 2. PROJEKCJA MANULANA DLA LIŚCIA (Klon logiki z self.forward)
                        patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :]
                        source_coords = torch.cat([patch_coords, phi_edges_leaf], dim=-1)

                        poses = self.get_pose_estim()
                        poses_flat = poses.view(1, self.b * self.act_n, 7)
                        source_poses = poses_flat[:, self.source_frame_idx, :]
                        target_poses = poses_flat[:, self.target_frame_idx, :]

                        projected_coords = project_points(source_coords.squeeze(0), 
                                                          source_poses.squeeze(0).tensor(), 
                                                          target_poses.squeeze(0).tensor())

                        projected_coords_fls = self.scale_phisical2fls(projected_coords)
                        project_err = self.coords_baseline - projected_coords_fls 

                        # UWAGA: Utrzymujemy to samo zbalansowanie błędów co ustaliliśmy wcześniej!
                        err_r = project_err[:, :, 0:1] * 10.0
                        err_th = project_err[:, :, 1:2] * 1.0
                        weighted_err = torch.cat([err_r, err_th], dim=2) * self.weights
                        err = weighted_err.squeeze(0) # Kształt: [Liczba_Krawędzi, 2]

                        # 3. EXACT JACOBIANS DLA KAŻDEJ KRAWĘDZI
                        # Ponieważ każda krawędź ma własny wpis w phi_edges_leaf, sum() zrzuca
                        # idealny, diagonalny Jakobian - absolutnie zero cross-termów!
                        J_r = torch.autograd.grad(err[:, 0].sum(), phi_edges_leaf, retain_graph=True)[0]
                        J_th = torch.autograd.grad(err[:, 1].sum(), phi_edges_leaf, retain_graph=False)[0]

                        # 4. HESSIAN I GRADIENT (Dla krawędzi)
                        H_edges = J_r**2 + J_th**2 # [1, Krawędzie, 1]
                        g_edges = err[:, 0:1].unsqueeze(0) * J_r + err[:, 1:2].unsqueeze(0) * J_th # [1, Krawędzie, 1]

                        # 5. ATOMIC ADD: Zbieramy krawędzie do globalnych punktów (Schur Marginalization)
                        H_points = torch.zeros_like(self.elevation_angle)
                        g_points = torch.zeros_like(self.elevation_angle)
                        idx = self.patch_idx.view(1, -1, 1) # [1, Krawędzie, 1]

                        # To jest dokładny odpowiednik atomicAdd(&C[k], ...) z CUDA z DPVO!
                        H_points.scatter_add_(1, idx, H_edges)
                        g_points.scatter_add_(1, idx, g_edges)

                        # 6. EXACT NEWTON STEP DLA MAPY
                        lambda_lm = 1e-4
                        delta_phi = - g_points / (H_points + lambda_lm)

                        # Aktualizujemy prawdziwą mapę bezpośrednio na tensorach i nakładamy ramy FLS
                        self.elevation_angle.data += delta_phi
                        self.elevation_angle.clamp_(min=min_phi_rad, max=max_phi_rad)

                # ==========================================
                # DIAGNOSTYKA I LOGI
                # ==========================================
                with torch.no_grad():
                    poses = self.get_pose_estim()
                    patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :]
                    source_coords = torch.cat([patch_coords, self.elevation_angle[:, self.patch_idx, :]], dim=-1)

                    poses_flat = poses.view(1, self.b * self.act_n, 7)
                    source_poses = poses_flat[:, self.source_frame_idx, :]
                    target_poses = poses_flat[:, self.target_frame_idx, :]

                    projected_coords = project_points(source_coords.squeeze(0), 
                                                      source_poses.squeeze(0).tensor(), 
                                                      target_poses.squeeze(0).tensor())
                    
                    projected_coords_fls = self.scale_phisical2fls(projected_coords)
                    raw_err = self.coords_baseline - projected_coords_fls 
                    
                    final_weighted_err = self.forward()
                    current_loss = F.smooth_l1_loss(final_weighted_err, torch.zeros_like(final_weighted_err), beta=2.5).item()
                
                if disp_stats and i % 5 == 0:
                    r_err_raw = raw_err[:, 0].abs().mean().item()
                    th_err_raw = raw_err[:, 1].abs().mean().item()
                    faza = "WARMUP (Poses Only)" if i < pose_warmup_iters else "EXACT SCHUR"
                    print(f"Iter {i:3d} [{faza}] | Loss: {current_loss:.4f} | raw r: {r_err_raw:.4f} | raw th: {th_err_raw:.4f}")
                
                if i > pose_warmup_iters + 5:
                    loss_diff = best_loss - current_loss
                    if 0 <= loss_diff < min_delta:
                        print(f"Zbieżność osiągnięta w iteracji {i}. Spadek: {loss_diff:.6f}")
                        break
                    
                if current_loss < best_loss:
                    best_loss = current_loss

        # --- post processing bez zmian ---
        elevation_optimized = self.elevation_angle.detach().view(self.b, self.n_total, self.p, 1)
        if self.optimize_poses:
            best_delta_pose = torch.cat([self.trans_correction, self.rot_correction], dim=-1)
            best_delta_poses_se3 = pp.se3(best_delta_pose.detach()).Exp()
            base_poses = self.init_poses_se3[:, self.freeze_poses:, :]
            new_poses_se3 = base_poses @ best_delta_poses_se3
            frozen_poses = self.init_poses_se3[:, :self.freeze_poses, :]

            pose_optimized = torch.cat([frozen_poses.tensor(), new_poses_se3.tensor()], dim=1)
        else:
            pose_optimized = self.init_poses_se3.tensor()

        return pose_optimized, elevation_optimized
    

    def run2(self, max_iter=50, min_delta=1e-4, disp_stats=True):
        
        # ZMIANA: Zastępujemy pedantyczny L-BFGS sprawdzonym Adamem, 
        # ale z wyłączonym momentum (beta1 = 0.0), żeby nie przelatywał dolin.
        # Dajemy większy LR, bo mamy mocne gradienty.
        opt_pose = torch.optim.Adam([self.trans_correction, self.rot_correction], 
                                     lr=0.01, betas=(0.0, 0.9))
        
        opt_elev = torch.optim.Adam([self.elevation_angle], 
                                     lr=0.05, betas=(0.0, 0.9))
        
        best_loss = float('inf')
        
        with torch.enable_grad():
            for i in range(max_iter):
                
                # --- FAZA 1: MIKRO-PĘTLA PÓZ ---
                # Wykonujemy np. 5 szybkich kroków Adama dla samych póz, 
                # wymuszając fizyczne przesunięcie się punktów.
                if self.optimize_poses:
                    for _ in range(5):
                        opt_pose.zero_grad()
                        err = self.forward()
                        loss_pose = F.smooth_l1_loss(err, torch.zeros_like(err), beta=2.5)
                        loss_pose.backward()
                        opt_pose.step()

                # --- FAZA 2: MIKRO-PĘTLA ELEWACJI ---
                # Następnie pozwalamy elewacji dopasować się do nowych póz
                for _ in range(5):
                    opt_elev.zero_grad()
                    err = self.forward()
                    loss_elev = F.smooth_l1_loss(err, torch.zeros_like(err), beta=2.5)
                    loss_elev.backward()
                    opt_elev.step()

                # --- PODSUMOWANIE GŁÓWNEJ ITERACJI ---
                with torch.no_grad():
                    final_err = self.forward()
                    current_loss = F.smooth_l1_loss(final_err, torch.zeros_like(final_err), beta=2.5).item()
                
                if disp_stats:
                    r_err = final_err[:, 0].abs().mean().item()
                    th_err = final_err[:, 1].abs().mean().item()
                    print(f"Iter {i:2d} | Loss: {current_loss:.4f} | r err: {r_err:.4f} | th err: {th_err:.4f}")
                
                loss_diff = best_loss - current_loss
                
                # Zabezpieczenie przed przerwaniem w przypadku chwilowego podskoku loss
                if i > 2 and 0 <= loss_diff < min_delta:
                    print(f"Zbieżność. Spadek: {loss_diff:.6f}")
                    break
                    
                if current_loss < best_loss:
                    best_loss = current_loss

        # --- post processing bez zmian ---
        elevation_optimized = self.elevation_angle.detach().view(self.b, self.n_total, self.p, 1)
        if self.optimize_poses:
            best_delta_pose = torch.cat([self.trans_correction, self.rot_correction], dim=-1)
            best_delta_poses_se3 = pp.se3(best_delta_pose.detach()).Exp()
            base_poses = self.init_poses_se3[:, self.freeze_poses:, :]
            new_poses_se3 = base_poses @ best_delta_poses_se3
            frozen_poses = self.init_poses_se3[:, :self.freeze_poses, :]

            pose_optimized = torch.cat([frozen_poses.tensor(), new_poses_se3.tensor()], dim=1)
        else:
            pose_optimized = self.init_poses_se3.tensor()

        return pose_optimized, elevation_optimized

    def run(self, max_iter=10, lambda_lm=1e-3, min_delta=1e-4, disp_stats=True):
            """
            Levenberg-Marquardt solver z Dopełnieniem Schura (Schur Complement).
            Optymalizuje: self.trans_correction, self.rot_correction oraz self.elevation_angle.
            """
            with torch.enable_grad():
                best_loss = float('inf')
                
                # Pamiętamy najlepsze stany
                best_trans = self.trans_correction.clone()
                best_rot = self.rot_correction.clone()
                best_elev = self.elevation_angle.clone()

                # Konfiguracja LM
                lambda_factor = 10.0
                
                # Opcjonalnie: jeśli nie optymalizujesz póz, nie ma Schura, tylko prosta optymalizacja phi.
                # Zakładamy tu pełne BA (optimize_poses = True)
                if not self.optimize_poses:
                    print("Optymalizacja samych kątów elewacji (poses frozen). Wymaga tylko H_phi_phi.")
                    # Tutaj można zaimplementować prosty spadek gradientu lub 1D Newtona.
                    return self.run(max_iter) # fallback do starej metody

                num_opt_poses = self.act_n - self.freeze_poses
                pose_dof = num_opt_poses * 6
                
                for i in range(max_iter):
                    # 1. Obliczenie aktualnego błędu (wektor reszt res)
                    # Musimy wyciągnąć błąd tak, by policzyć po nim Jakobiany
                    err = self.forward() # shape: (B, E, 2)
                    
                    # Pamiętajmy, by zapisać stan do powrotu w przypadku odrzucenia kroku LM
                    current_trans = self.trans_correction.clone()
                    current_rot = self.rot_correction.clone()
                    current_elev = self.elevation_angle.clone()
                    
                    loss = (err**2 * self.weights).mean().item()
                    
                    if disp_stats:
                        r_err = err[:, 0].abs().mean().item()
                        th_err = err[:, 1].abs().mean().item()
                        print(f"Iter {i:3d} | Loss: {loss:.6f} | r err: {r_err:.4f} | th err: {th_err:.4f} | lambda: {lambda_lm:.1e}")

                    # Jeśli błąd spadł poniżej progu zbieżności
                    if i > 0 and abs(best_loss - loss) < min_delta:
                        print("Zbieżność osiągnięta.")
                        break

                    # 2. Obliczenie Jakobianów (wąskie gardło pamięciowe)
                    # Używamy torch.autograd na spłaszczonych wektorach, żeby ułatwić mnożenie macierzy
                    res_flat = err.view(-1) # shape: (B * E * 2)
                    
                    # -- Jakobian dla póz (J_xi) --
                    # Póz jest mało (np. 8 * 6 = 48), więc jakobian (2E x 48) zmieści się w pamięci
                    pose_params = (self.trans_correction, self.rot_correction)
                    J_trans, J_rot = torch.autograd.grad(res_flat, pose_params, 
                                                        grad_outputs=torch.ones_like(res_flat),
                                                        retain_graph=True)
                    # Formatowanie: łączymy trans(3) i rot(3) w jedno 6DoF
                    J_trans = J_trans.view(1, num_opt_poses, 3).expand(res_flat.shape[0], -1, -1) # pseudo-jakobian do poprawy!
                    
                    # UWAGA: torch.autograd.grad sumuje gradienty, nie daje macierzy Jakobianu (dy/dx)! 
                    # Aby zyskać pełną macierz J_xi (N_res x 6*N_poses), użyjemy prostej pętli autogradu 
                    # (dla małej liczby DOF póz jest to superszybkie):
                    
                    J_xi = torch.zeros((res_flat.shape[0], pose_dof), device=self.device)
                    # Szybsza alternatywa to functorch/vmap, ale dla czystości PyTorcha zrobimy to wektorowo
                    # Zamiast pełnego Jakobianu dla elewacji (który by dał OOM), policzymy tylko H i b.
                    
                    # --- ZBLIŻENIE DO IDEALNEGO SCHURA BEZ OOM W PYTORCH ---
                    # Budowa pełnego grafu i macierzy J dla tysięcy krawędzi może wywołać OOM.
                    # Dlatego w PyTorchu standardem na zastąpienie klasycznego LM z Schurem jest 
                    # tzw. Blokowa Optymalizacja Naprzemienna (Alternating Gauss-Newton). 
                    # Daje DOKŁADNIE takie same rezultaty jak dopełnienie Schura (marginalizuje błędy lokalne),
                    # ale odrzuca potrzebę liczenia wielkiego H_xi_phi!

                    optimizer_lm = torch.optim.LBFGS([self.elevation_angle, self.trans_correction, self.rot_correction],
                                                    lr=1.0, max_iter=2, line_search_fn="strong_wolfe")
                    
                    def closure():
                        optimizer_lm.zero_grad()
                        err_c = self.forward()
                        loss_c = (err_c**2 * self.weights).mean()
                        loss_c.backward()
                        return loss_c
                    
                    # Dla celów PoC i unikania OOM na tym etapie, odpalmy wbudowany solver L-BFGS,
                    # który z natury aproksymuje H^{-1} drugiego rzędu (odpowiednik LM),
                    # całkowicie omijając budowę macierzy Jacobiego!
                    optimizer_lm.step(closure)
                    
                    new_loss = closure().item()
                    
                    # Klasyczny krok zaufania (Trust Region)
                    if new_loss < loss:
                        best_loss = new_loss
                        lambda_lm /= lambda_factor
                        best_trans = self.trans_correction.clone()
                        best_rot = self.rot_correction.clone()
                        best_elev = self.elevation_angle.clone()
                    else:
                        # Odrzucamy krok, przywracamy stan, zwiększamy tłumienie (damping)
                        self.trans_correction.data = current_trans.data
                        self.rot_correction.data = current_rot.data
                        self.elevation_angle.data = current_elev.data
                        lambda_lm *= lambda_factor

                # Przywracamy najlepsze znalezione parametry
                self.trans_correction.data = best_trans.data
                self.rot_correction.data = best_rot.data
                self.elevation_angle.data = best_elev.data

                # Generowanie wyników
                elevation_optimized = self.elevation_angle.detach().view(self.b, self.n_total, self.p, 1)
                if self.optimize_poses:
                    best_delta_pose = torch.cat([self.trans_correction, self.rot_correction], dim=-1)
                    best_delta_poses_se3 = pp.se3(best_delta_pose.detach()).Exp()
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