import numpy as np
import torch
import torch.nn.functional as F
import pypose as pp
from scipy.spatial.transform import Rotation as R


def eval_metrics_3d(pred, gt, reduction = 'mean'):

    # torch + pypose

    b, n, _ = pred.shape
    pred = pred.view(b*n, -1)
    gt = gt[:, :n, :] # if less pred poses than gt
    gt = gt.view(b*n, -1)
    
    # create SE3 objects
    pred_se3 = pp.SE3(pred)
    gt_se3 = pp.SE3(gt)

    # --- ABSOLUTE ERROR --- 
    absolute_pose_error_unreduced = (pred_se3.Inv() @ gt_se3)
    # Absolute trajectory error - unreduced
    ATE = absolute_pose_error_unreduced.translation().norm(dim=-1)
    # Absolute rotation error - unreduced
    ARE = absolute_pose_error_unreduced.Log()[:, 3:].norm(dim=-1)
    
    
    # --- RELATIVE ERROR --- 
    pred_diff_se3 = pred_se3[:-1, :].Inv() * pred_se3[1:, :]
    gt_diff_se3 = gt_se3[:-1, :].Inv() * gt_se3[1:, :]
    diff_se3 = pred_diff_se3.Inv() * gt_diff_se3

    # Relative trajectory error - unreduced
    RPE_trans = diff_se3.translation().norm(dim=-1)
    # Relative rotation error - unreduced
    RPE_rot = diff_se3.Log()[:, 3:].norm(dim=-1)
    
    if reduction == 'mean':
        metrics = {
            'ATE': torch.mean(ATE),
            'ARE': torch.mean(ARE),
            'RPE_translation': torch.mean(RPE_trans),
            'RPE_rotation': torch.mean(RPE_rot)
        }
    # elif reduction == 'mse':
    #     metrics = {
    #         'ATE': F.mse_loss(ATE),
    #         'ARE': F.mse_loss(ARE),
    #         'RPE_translation': F.mse_loss(RPE_trans),
    #         'RPE_rotation': F.mse_loss(RPE_rot)
    #     }
    else:
        metrics = {
            'ATE': ATE,
            'ARE': ARE,
            'RPE_translation': RPE_trans,
            'RPE_rotation': RPE_rot,
            'num_steps': b*n
        }

    return metrics 




def eval_metrics_2d(pred, gt):
    """
    Oblicza i wyświetla metryki odometrii 2D (translacja + rotacja) 
    w standardzie publikacji naukowych.
    
    pred: numpy array (N, 3) -> [x, y, theta]
    gt: numpy array (N, 3) -> [x, y, theta]
    """
    # --- Rozdzielenie translacji i rotacji ---
    pred_xy, pred_theta = pred[:, :2], pred[:, 2]
    gt_xy, gt_theta = gt[:, :2], gt[:, 2]

    # Bezpieczna różnica kątowa eliminująca problem przeskoku na -pi/pi
    def norm_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    # ==========================================
    # 1. TRANSLACJA (Translational Metrics)
    # ==========================================
    step_distances_gt = np.linalg.norm((gt_xy[1:] - gt_xy[:-1]), axis=1)
    total_distance = np.sum(step_distances_gt)
    
    # Absolute Trajectory Error (ATE)
    abs_traj_l2 = np.linalg.norm((gt_xy - pred_xy), axis=1)
    ate_rmse = np.sqrt(np.mean(abs_traj_l2**2))
    ate_perc = (ate_rmse / total_distance) * 100 if total_distance > 0 else 0.0
    
    # Final Position Drift
    final_drift_m = abs_traj_l2[-1]
    final_drift_perc = (final_drift_m / total_distance) * 100 if total_distance > 0 else 0.0

    # Relative Translation Error (RTE)
    relative_step_pred = pred_xy[1:] - pred_xy[:-1]
    relative_step_gt = gt_xy[1:] - gt_xy[:-1]
    
    rel_traj_l2 = np.linalg.norm((relative_step_gt - relative_step_pred), axis=1)
    rte_rmse = np.sqrt(np.mean(rel_traj_l2**2))
    rte_step_perc = np.mean(rel_traj_l2 / (step_distances_gt + 1e-8)) * 100

    # ==========================================
    # 2. ROTACJA (Rotational Metrics)
    # ==========================================
    # Całkowity obrót (suma bezwzględnych kroków kątowych)
    step_rotations_gt = np.abs(norm_angle(gt_theta[1:] - gt_theta[:-1]))
    total_rotation = np.sum(step_rotations_gt)
    
    # Absolute Rotation Error (ARE)
    abs_rot_err = np.abs(norm_angle(gt_theta - pred_theta))
    are_rmse = np.sqrt(np.mean(abs_rot_err**2))
    are_perc = (are_rmse / total_rotation) * 100 if total_rotation > 0 else 0.0
    
    # Final Rotation Drift
    final_rot_drift = abs_rot_err[-1]
    final_rot_perc = (final_rot_drift / total_rotation) * 100 if total_rotation > 0 else 0.0
    
    # Relative Rotation Error (RRE)
    rel_step_theta_pred = norm_angle(pred_theta[1:] - pred_theta[:-1])
    rel_step_theta_gt = norm_angle(gt_theta[1:] - gt_theta[:-1])
    
    rel_rot_err = np.abs(norm_angle(rel_step_theta_gt - rel_step_theta_pred))
    rre_rmse = np.sqrt(np.mean(rel_rot_err**2))
    rre_step_perc = np.mean(rel_rot_err / (step_rotations_gt + 1e-8)) * 100


    print("=" * 80)
    print(f"{'Odometry Evaluation Metrics (Translation & Rotation)':^80}")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Absolute':>16} | {'Relative [%]':>15}")
    print("-" * 80)
    print(f"{'Total Trajectory Length':<35} | {total_distance:>14.4f} m | {'-':>15} ")
    print(f"{'Total Rotation Length':<35} | {total_rotation:>12.4f} rad | {'-':>15} ")
    print("-" * 80)
    print(f"{'Absolute Trajectory Err (ATE)':<35} | {ate_rmse:>14.4f} m | {ate_perc:>14.4f} %")
    print(f"{'Final Position Drift':<35} | {final_drift_m:>14.4f} m | {final_drift_perc:>14.4f} %")
    print(f"{'Relative Translation Err (RTE)':<35} | {rte_rmse:>14.4f} m | {rte_step_perc:>14.4f} %")
    print("-" * 80)
    print(f"{'Absolute Rotation Err (ARE)':<35} | {are_rmse:>12.4f} rad | {are_perc:>14.4f} %")
    print(f"{'Final Rotation Drift':<35} | {final_rot_drift:>12.4f} rad | {final_rot_perc:>14.4f} %")
    print(f"{'Relative Rotation Err (RRE)':<35} | {rre_rmse:>12.4f} rad | {rre_step_perc:>14.4f} %")
    print("=" * 80)

    metrics = {
        "distance": total_distance,
        "rotation": total_rotation,
        
        "ate_rmse": ate_rmse,
        "ate_perc": ate_perc,
        "final_drift_m": final_drift_m,
        "final_drift_perc": final_drift_perc,
        "rte_rmse": rte_rmse,
        "rte_perc": rte_step_perc,
        
        "are_rmse": are_rmse,
        "are_perc": are_perc,
        "final_rot_drift": final_rot_drift,
        "final_rot_perc": final_rot_perc,
        "rre_rmse": rre_rmse,
        "rre_perc": rre_step_perc
    }
    
    return metrics

