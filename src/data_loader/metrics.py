import numpy as np
import torch
import torch.nn.functional as F
import pypose as pp
from scipy.spatial.transform import Rotation as R


def eval_metrics(pred, gt, reduction = 'mean'):

    b, n, _ = pred.shape
    pred = pred.view(b*n, -1)
    gt = gt[:, :n, :] # if less pred poses than gt
    gt = gt.view(b*n, -1)
    

    # create SE3 objects
    pred_se3 = pp.SE3(pred)
    gt_se3 = pp.SE3(gt)

    # --- ABSOLUTE ERROR --- 
    absolute_pose_error_unreduced = (pred_se3.Inv() * gt_se3)
    # Absolute trajectory error - unreduced
    ATE = absolute_pose_error_unreduced.translation().norm()
    # Absolute rotation error - unreduced
    ARE = absolute_pose_error_unreduced.Log()[:, 3:].norm(dim-1)
    
    
    # --- RELATIVE ERROR --- 
    pred_diff_se3 = pred_se3[:-1, :].Inv() * pred_se3[1:, :]
    gt_diff_se3 = gt_se3[:-1, :].Inv() * gt_se3[1:, :]
    diff_se3 = pred_diff_se3.Inv() * gt_diff_se3

    # Relative trajectory error - unreduced
    RPE_trans = diff_se3.translation().norm()
    # Relative rotation error - unreduced
    RPE_rot = diff_se3.Log()[:, 3:].norm(dim=-1)
    
    if reduction == 'mean':
        metrics = {
            'ATE': torch.mean(ATE),
            'ARE': torch.mean(ARE),
            'RPE_translation': torch.mean(RPE_trans),
            'RPE_rotation': torch.mean(RPE_rot)
        }
    else:
        metrics = {
            'ATE': ATE,
            'ARE': ARE,
            'RPE_translation': RPE_trans,
            'RPE_rotation': RPE_rot,
            'num_steps': b*n
        }

    return metrics 
    






# # === Training metrics - PyTorch === 

# def translation_err(x_pred, x_target):
#     '''
#     Distance betweend two poses, q = (x, y, z).
#     Euclidesian norm (L2) from vector difference.
#     dist_err = L2(q1 - q2)
#     '''
#     Lx = torch.linalg.norm(x_target-x_pred, dim=1)
#     return Lx
 
# def rot_err(q_pred, q_target):
#     '''
#     Rotation error:
#     Angle $\theta$ extracted from difference quaterion $\Delta q$:
#     $$\Delta q = q_{pred}^{-1} \otimes q_{target}$$.

#     If only rotation angle is needed, this problem can be simoplified to:
#     - calculatation of qw quaternion component. 
#     - extraction $\theta$ from quaternion construction: w = cos($0.5*\theta$)
#     '''
#     # find shortest rotation 
#     dot = (q_pred * q_target).sum(dim=-1, keepdim=True) # real part of difference quaternion -> dot product
#     q_dist = torch.abs(dot) # shortest rotation
#     q_dist = torch.clamp(q_dist, max=1.0 - 1e-7)

#     return 2*torch.arccos(q_dist)

# def odometry_evaluation_metrics(pred, target, reduction = 'mean'):
    
#     b, n, _ = pred.shape
    
#     target_act = target[:, :n, :] # if less pred poses than gt
#     target_act = target_act.view(b*n, -1)
    
#     pred = pred.view(b*n, -1)
   

#     x_pred, x_target = pred[:, :3], target_act[:, :3] # translation
#     q_pred, q_target = pred[:, 3:7], target_act[:, 3:7] # quaternion

    

#     # --- Absolute errors ---

#     # --- absolut translation --- 
#     abs_dist_error_unreduced = dist_err(x_pred, x_target) # L2 
    
#     # --- absolute rotation --- 
#     abs_rot_error_unreduced = rot_err(q_pred, q_target) # relative rotatio nangl

#     # --- Relative errors ---

#     # --- relative translation --- 
#     x_pred_rel = x_pred[1:, :] - x_pred[:-1, :]
#     x_target_rel = x_target[1:, :] - x_target[:-1, :]

#     dist_rel_error_unreduced = Lx = torch.linalg.norm(x_target_rel - x_pred_rel, dim=1)
    
#     # --- relative rotation ---

#     q_pred_rel = torch.sum(q_pred[1:, :] * q_pred[:-1, :], dim =-1)
#     q_target_rel = torch.sum(q_target[1:, :] * q_target[:-1, :], dim =-1)

#     w_pred_rel = torch.clamp(torch.abs(q_pred_rel), 1 - 1e-7)
#     w_target_rel =  torch.clamp(torch.abs(q_target_rel), 1 - 1e-7)

#     rel_pred_angle = 2*torch.arccos(w_pred_rel)
#     rel_target_angle = 2*torch.arccos(w_pred_rel)

#     mean_rel_rot_err_unreduced = rel_target_angle - rel_pred_angle

#     # --- metrics dict ---
#     if reduction == 'mean':
#         metrics = {'ATE': torch.mean(abs_dist_error_unreduced), 
#                    'RPE': torch.mean(dist_rel_error_unreduced), 
#                    'MEAN_ABS_ROT_ERR': torch.mean(torch.abs(abs_rot_error_unreduced)),
#                    'MEAN_REL_ROT_ERR':torch.mean(torch.abs(mean_rel_rot_err_unreduced))
#                   }
#     else: 
#          metrics = {'ATE': abs_dist_error_unreduced, 
#                     'RPE': dist_rel_error_unreduced, 
#                     'MEAN_ABS_ROT_ERR': abs_rot_error_unreduced,
#                     'MEAN_REL_ROT_ERR': rel_target_angle - rel_pred_angle,
#                   }
    
#     return ATE, RPE, mean_abs_rot_err, mean_rel_rot_err










# def pose_err(pred, target):

#     b, n, _ = pred.shape
#     target_act = target[:, :n, :]
#     pred = pred.view(b*n, -1)
#     target_act = target_act.view(b*n, -1)

#     x_pred, x_target = pred[:, :3], target_act[:, :3]
#     q_pred, q_target = pred[:, 3:7], target_act[:, 3:7]

#     dist = dist_err(x_pred, x_target)
#     rot = rot_err(q_pred, q_target)
    
#     return torch.mean(dist), torch.mean(rot)















# # === EVALUATION METRICS - numpy ===

# def align_traj(pred, target, init_pt_only=False):
#     trans_pred = pred[:, :3]
#     trans_target = target[:, :3]
#     q_pred = pred[:, 3:]
#     q_target = target[:, 3:]

#     # Center trajectory
#     if init_pt_only:
#         centroid_pred = np.mean(trans_pred[:2, :], axis=0)
#         centroid_target = np.mean(trans_target[:2, :], axis=0)
#     else:
#         centroid_pred = np.mean(trans_pred, axis=0)
#         centroid_target = np.mean(trans_target, axis=0)
    
#     pred_centered = trans_pred - centroid_pred
#     target_centered = trans_target - centroid_target

#     # Rotation alignment
#     if init_pt_only:
#         target_align_base = target_centered[:2, :]
#         pred_align_base = pred_centered[:2, :]
#     else: 
#         target_align_base = target_centered
#         pred_align_base = pred_centered

#     global_rot, _ = R.align_vectors(target_align_base, pred_align_base)

#     # Translation alignment
#     global_trans = centroid_target - global_rot.apply(centroid_pred)  
#     trans_pred_aligned = global_rot.apply(trans_pred) + global_trans   

#     # Rotation alignment for quaternions
#     r = R.from_quat(q_pred)
#     r_pred_aligned = global_rot * r
#     q_pred_aligned = r_pred_aligned.as_quat()

#     pred_aligned = np.concatenate([trans_pred_aligned, q_pred_aligned], axis=1)
#     return pred_aligned


# def RPE(pred, target, delta=1):
#     n = pred.shape[0] - delta
#     if n <= 0:
#         return np.array([]), 0.0

#     Q_i_t = pred[:-delta, :3] 
#     Q_i_r = R.from_quat(pred[:-delta, 3:]) 
#     Q_d_t = pred[delta:, :3] 
    
#     P_i_t = target[:-delta, :3] 
#     P_i_r = R.from_quat(target[:-delta, 3:]) 
#     P_d_t = target[delta:, :3] 

#     # Relative position
#     Q_rel_pos = Q_i_r.inv().apply(Q_d_t - Q_i_t)
#     P_rel_pos = P_i_r.inv().apply(P_d_t - P_i_t)

#     # Translation error norms for each pair
#     E = np.linalg.norm(P_rel_pos - Q_rel_pos, axis=1)

#     # RMSE over all pairs
#     rmse_rpe = np.sqrt(np.mean(E**2))

#     return E, rmse_rpe


# def ATE(pred, target):
    
#     x_pred, x_target = pred[:, :3], target[:, :3]
#     q_pred, q_target = pred[:, 3:7], target[:, 3:7]

#     # Translation error
#     dist = np.linalg.norm(x_target - x_pred, axis=1)

#     # Rotation error
#     dot = np.sum(q_pred * q_target, axis=1)
#     q_dist = np.abs(dot)
#     q_dist = np.clip(q_dist, a_min=None, a_max=1.0 - 1e-7) 
#     rot = 2 * np.arccos(q_dist)

#     # RMSE
#     rmse_ate = np.sqrt(np.mean(dist**2))
#     rmse_rot = np.sqrt(np.mean(rot**2))

#     return rmse_ate, rmse_rot, dist, rot


# # def eval_metrics(pred, target, align=True, align_init_pt_only=True, add_data_series=False):
  
# #     n = min(pred.shape[0], target.shape[0])
# #     pred = pred[:n, :]
# #     target = target[:n, :]

# #     if align:
# #         pred_align = align_traj(pred, target, init_pt_only=align_init_pt_only)
# #     else:
# #         pred_align = pred

# #     # Absolute trajectory error
# #     rmse_ate, rmse_rot, dist, rot = ATE(pred_align, target)

# #     # Relative pose error 
# #     vect_rpe, rmse_rpe = RPE(pred, target)
 
# #     metrics = {
# #         # Global metrics
# #         'RMSE_ATE': float(rmse_ate),
# #         'RMSE_RPE': float(rmse_rpe),
# #         'RMSE_ROT': float(rmse_rot),
        
# #         # Translation stats 
# #         'MEAN_TRANS_ERR': float(np.mean(dist)),
# #         'MEDIAN_TRANS_ERR': float(np.median(dist)),
# #         'STD_TRANS_ERR': float(np.std(dist)),
# #         'MIN_TRANS_ERR': float(np.min(dist)),
# #         'MAX_TRANS_ERR': float(np.max(dist)),
        
# #         # Rotation stats
# #         'MEAN_ROT_ERR': float(np.mean(rot)),
# #         'MEDIAN_ROT_ERR': float(np.median(rot)),
# #         'STD_ROT_ERR': float(np.std(rot)),
# #         'MIN_ROT_ERR': float(np.min(rot)),
# #         'MAX_ROT_ERR': float(np.max(rot))

# #     }
      

# #     if add_data_series:
# #         metrics['data_absolute_translation'] = dist
# #         metrics['data_relative_translation'] = vect_rpe
# #         metrics['data_absolute_rotation'] = rot
# #     return metrics

# def eval_metrics(pred, target, align=True, align_init_pt_only=True, add_data_series=False):
    
#     if pred.ndim == 2:
#         pred = np.expand_dims(pred, axis=0)
#         target = np.expand_dims(target, axis=0)

#     B, _, _ = pred.shape
#     n_act = min(pred.shape[1], target.shape[1])

#     all_dist = []
#     all_rot = []
#     all_vect_rpe = []

#     # iterate over batch
#     for b in range(B):
#         p = pred[b, :n_act, :]
#         t = target[b, :n_act, :]

#         if align:
#             p_align = align_traj(p, t, init_pt_only=align_init_pt_only)
#         else:
#             p_align = p

#         # Absolute trajectory error
#         _, _, dist, rot = ATE(p_align, t)
#         all_dist.append(dist)
#         all_rot.append(rot)

#         # Relative pose error 
#         vect_rpe, _ = RPE(p, t)
#         if vect_rpe.size > 0:
#             all_vect_rpe.append(vect_rpe)

#     dist_cat = np.concatenate(all_dist)
#     rot_cat = np.concatenate(all_rot)
    
#     if len(all_vect_rpe) > 0:
#         vect_rpe_cat = np.concatenate(all_vect_rpe)
#         rmse_rpe_overall = np.sqrt(np.mean(vect_rpe_cat**2))
#     else:
#         vect_rpe_cat = np.array([])
#         rmse_rpe_overall = 0.0

#     rmse_ate_overall = np.sqrt(np.mean(dist_cat**2))
#     rmse_rot_overall = np.sqrt(np.mean(rot_cat**2))

#     metrics = {
#         'RMSE_ATE': float(rmse_ate_overall),
#         'RMSE_RPE': float(rmse_rpe_overall),
#         'RMSE_ROT': float(rmse_rot_overall),
        
#         'MEAN_TRANS_ERR': float(np.mean(dist_cat)),
#         'MEDIAN_TRANS_ERR': float(np.median(dist_cat)),
#         'STD_TRANS_ERR': float(np.std(dist_cat)),
#         'MIN_TRANS_ERR': float(np.min(dist_cat)),
#         'MAX_TRANS_ERR': float(np.max(dist_cat)),
        
#         'MEAN_ROT_ERR': float(np.mean(rot_cat)),
#         'MEDIAN_ROT_ERR': float(np.median(rot_cat)),
#         'STD_ROT_ERR': float(np.std(rot_cat)),
#         'MIN_ROT_ERR': float(np.min(rot_cat)),
#         'MAX_ROT_ERR': float(np.max(rot_cat))
#     }
      
#     if add_data_series:
#         metrics['data_absolute_translation'] = dist_cat
#         metrics['data_relative_translation'] = vect_rpe_cat
#         metrics['data_absolute_rotation'] = rot_cat
        
#     return metrics


# # b = 1  # batch size
# # n = 5  # liczba aktywnych póz w danym kroku (np. dotarłeś do klatki nr 4, więc masz 5 póz)

# # # Generujemy losowe dane o kształcie [B, N, 7]
# # pred_poses = torch.rand(b, n, 7)
# # target_poses = torch.rand(b, n, 7)

# # # Normalizacja kwaternionów (żeby było zgodnie ze sztuką)
# # pred_poses[:, :, 3:7] = torch.nn.functional.normalize(pred_poses[:, :, 3:7], p=2, dim=-1)
# # target_poses[:, :, 3:7] = torch.nn.functional.normalize(target_poses[:, :, 3:7], p=2, dim=-1)

# # # Puszczamy przez Twoją funkcję
# # dist_loss, rot_loss = pose_err(pred_poses, target_poses)

# # print(f"Kształt wejścia (pred_poses): {pred_poses.shape}")
# # print("-" * 40)
# # print(f"Kształt wyjścia dist_loss:    {dist_loss.shape}")
# # print(f"Kształt wyjścia rot_loss:     {rot_loss.shape}")
