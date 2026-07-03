import numpy as np
import torch
import torch.nn.functional as F
import pypose as pp
from scipy.spatial.transform import Rotation as R


def eval_metrics(pred, gt, reduction = 'mean'):

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
    Oblicza i wyświetla metryki odometrii w standardzie publikacji naukowych.
    pred: numpy array (N, 2)
    gt: numpy array (N, 2)
    """
    # 1. Obliczenie całkowitej długości trajektorii GT
    step_distances_gt = np.linalg.norm((gt[1:, :] - gt[:-1, :]), axis=1)
    total_distance = np.sum(step_distances_gt)
    
    # 2. Absolute Trajectory Error (ATE)
    abs_traj_l2 = np.linalg.norm((gt - pred), axis=1)
    ate_rmse = np.sqrt(np.mean(abs_traj_l2**2))
    ate_perc = (ate_rmse / total_distance) * 100 if total_distance > 0 else 0.0
    
    # 3. Final Drift (Błąd końcowej pozycji - kluczowe w nawigacji AUV)
    final_drift_m = abs_traj_l2[-1]
    final_drift_perc = (final_drift_m / total_distance) * 100 if total_distance > 0 else 0.0

    # 4. Relative Translation Error (RTE / RPE)
    relative_step_pred = pred[1:, :] - pred[:-1, :]
    relative_step_gt = gt[1:, :] - gt[:-1, :]
    
    rel_traj_l2 = np.linalg.norm((relative_step_gt - relative_step_pred), axis=1)
    rte_rmse = np.sqrt(np.mean(rel_traj_l2**2))
    
    # Błąd relatywny jako % długości kroku (zabezpieczenie przed dzieleniem przez zero)
    rte_step_perc = np.mean(rel_traj_l2 / (step_distances_gt + 1e-8)) * 100

    # --- Renderowanie tabeli ---
    print("=" * 68)
    print(f"{'Odometry Evaluation Metrics (SOTA format)':^68}")
    print("=" * 68)
    print(f"{'Metric':<30} | {'Absolute [m]':>14} | {'Relative [%]':>15}")
    print("-" * 68)
    print(f"{'Total Trajectory Length':<30} | {total_distance:>12.4f} m | {'-':>14} ")
    print(f"{'Absolute Trajectory Err (ATE)':<30} | {ate_rmse:>12.4f} m | {ate_perc:>13.4f} %")
    print(f"{'Final Position Drift':<30} | {final_drift_m:>12.4f} m | {final_drift_perc:>13.4f} %")
    print(f"{'Relative Translation Err (RTE)':<30} | {rte_rmse:>12.4f} m | {rte_step_perc:>13.4f} %")
    print("=" * 68)

    metrics = {
        "distance": total_distance,
        "ate_rmse": ate_rmse,
        "ate_perc": ate_perc,
        "final_drift_m": final_drift_m,
        "final_drift_perc": final_drift_perc,
        "rte_rmse": rte_rmse,
        "rte_perc": rte_step_perc
    }
    
    return metrics


# --- test ---

gt_test = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 0.0],
    [3.0, 0.0],
    [4.0, 0.0],
])
theta = np.deg2rad(5.0)  
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
t = np.array([1.0, 0.5])
pred_test = (gt_test @ R.T) + t

ATE, RPE = eval_metrics_2d(pred_test, gt_test)

print(f'Ate: {ATE}, expected')
