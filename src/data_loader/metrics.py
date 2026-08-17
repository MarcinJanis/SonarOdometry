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
    print(f"{'Odometry Evaluation Metrics':^80}")
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



from evo.core import trajectory, metrics
from evo.core.metrics import PoseRelation, Unit
import pandas as pd

def yaw_to_quaternion_wxyz(yaw_array: np.ndarray) -> np.ndarray:
    """Converts an array of yaw angles to quaternions [qw, qx, qy, qz]."""
    qz = np.sin(yaw_array / 2.0)
    qw = np.cos(yaw_array / 2.0)
    qx = np.zeros_like(yaw_array)
    qy = np.zeros_like(yaw_array)
    return np.column_stack([qw, qx, qy, qz])


def metrics_evo(df: pd.DataFrame, rpe_percent: float = 10.0) -> dict:
    """
    Takes a DataFrame containing odometry data, calculates APE and RPE metrics
    using the evo package, and prints a unified, neatly formatted evaluation table.
    
    Args:
        df: Pandas DataFrame with ground truth and predicted poses.
        rpe_percent: The segment length for RPE calculation, expressed as a 
                     percentage of the total Ground Truth trajectory length.
    
    Returns a dictionary with raw statistics.
    """
    # 1. Extract data from DataFrame
    timestamps = df["frame_id"].to_numpy(dtype=float)

    gt_xyz = np.column_stack([df["gt_x"], df["gt_y"], np.zeros(len(df))])
    pred_xyz = np.column_stack([df["pred_x"], df["pred_y"], np.zeros(len(df))])

    # Generate quaternions
    gt_quat = yaw_to_quaternion_wxyz(df["gt_theta"].to_numpy())
    pred_quat = yaw_to_quaternion_wxyz(df["pred_theta"].to_numpy())

    # 2. Calculate Total Distance & target RPE segment length
    step_distances = np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)
    total_distance = np.sum(step_distances)
    
    # Calculate distance equivalent to the requested percentage
    rpe_delta_meters = total_distance * (rpe_percent / 100.0)
    
    # Fallback to avoid division by zero in case of completely static trajectory
    if rpe_delta_meters <= 0.0:
        rpe_delta_meters = 1.0 

    # 3. Initialize trajectories
    gt_traj = trajectory.PoseTrajectory3D(
        positions_xyz=gt_xyz, 
        orientations_quat_wxyz=gt_quat, 
        timestamps=timestamps
    )
    
    pred_traj = trajectory.PoseTrajectory3D(
        positions_xyz=pred_xyz, 
        orientations_quat_wxyz=pred_quat, 
        timestamps=timestamps
    )

    # 4. Calculate APE (Absolute Pose Error) / ATE - Translation
    ape_trans = metrics.APE(PoseRelation.translation_part)
    ape_trans.process_data((gt_traj, pred_traj))
    ape_trans_stats = ape_trans.get_all_statistics()

    # 5. Calculate APE (Absolute Pose Error) - Rotation
    ape_rot = metrics.APE(PoseRelation.rotation_angle_rad)
    ape_rot.process_data((gt_traj, pred_traj))
    ape_rot_stats = ape_rot.get_all_statistics()

    # 6. Calculate RPE (Relative Pose Error) USING METERS instead of frames
    rpe_trans = metrics.RPE(
        PoseRelation.translation_part, 
        delta=rpe_delta_meters, 
        delta_unit=Unit.meters, 
        all_pairs=False
    )
    rpe_trans.process_data((gt_traj, pred_traj))
    rpe_trans_stats = rpe_trans.get_all_statistics()
    
    rpe_rot = metrics.RPE(
        PoseRelation.rotation_angle_rad, 
        delta=rpe_delta_meters, 
        delta_unit=Unit.meters, 
        all_pairs=False
    )
    rpe_rot.process_data((gt_traj, pred_traj))
    rpe_rot_stats = rpe_rot.get_all_statistics()

    # 7. Calculate KITTI-style relative errors (drift)
    # The error is accumulated over rpe_delta_meters, so we divide by that distance
    trans_error_percent = (rpe_trans_stats["mean"] / rpe_delta_meters) * 100.0

    rot_error_deg = np.degrees(rpe_rot_stats["mean"])
    rot_error_deg_per_m = rot_error_deg / rpe_delta_meters

    # ---------------------------------------------------------
    # DISPLAYING THE UNIFIED RESULTS TABLE
    # ---------------------------------------------------------
    table_width = 97
    
    print("=" * table_width)
    print(f"{'ODOMETRY EVALUATION METRICS':^{table_width}}")
    print("=" * table_width)
    print(f" Trajectory Length (GT) : {total_distance:.2f} m")
    print(f" RPE Evaluation Step    : {rpe_percent:.1f}% of trajectory length ({rpe_delta_meters:.2f} m)")
    print("-" * table_width)
    print(f"{'METRIC [UNIT]':<32} | {'RMSE':>8} | {'MEAN':>8} | {'MEDIAN':>8} | {'MIN':>8} | {'MAX':>8} | {'STD':>8}")
    print("-" * table_width)
    
    def print_row(name: str, stats: dict):
        print(f"{name:<32} | "
              f"{stats['rmse']:8.4f} | "
              f"{stats['mean']:8.4f} | "
              f"{stats['median']:8.4f} | "
              f"{stats['min']:8.4f} | "
              f"{stats['max']:8.4f} | "
              f"{stats['std']:8.4f}")
              
    def print_kitti_row(name: str, mean_val: float):
        print(f"{name:<32} | "
              f"{'---':>8} | "
              f"{mean_val:8.4f} | "
              f"{'---':>8} | "
              f"{'---':>8} | "
              f"{'---':>8} | "
              f"{'---':>8}")

    # Standard metrics
    print_row("APE Translation (ATE) [m]", ape_trans_stats)
    print_row("APE Rotation [rad]", ape_rot_stats)
    
    print_row(f"RPE Translation (d={rpe_delta_meters:.1f}m) [m]", rpe_trans_stats)
    print_row(f"RPE Rotation (d={rpe_delta_meters:.1f}m) [rad]", rpe_rot_stats)
    
    print("-" * table_width)
    print(f"{'RELATIVE DRIFT (KITTI STYLE)':^{table_width}}")
    print("-" * table_width)
    
    # KITTI-style metrics (using only the MEAN column)
    print_kitti_row(f"Translation Error [%]", trans_error_percent)
    print_kitti_row(f"Rotation Error [deg/m]", rot_error_deg_per_m)
    
    print("=" * table_width)

    return {
        "ape_trans": ape_trans_stats,
        "ape_rot": ape_rot_stats,
        "rpe_trans": rpe_trans_stats,
        "rpe_rot": rpe_rot_stats,
        "kitti_trans_pct": trans_error_percent,
        "kitti_rot_deg_per_m": rot_error_deg_per_m,
        "total_distance_m": total_distance,
        "rpe_segment_meters": rpe_delta_meters,
        "rpe_segment_percent": rpe_percent
    }