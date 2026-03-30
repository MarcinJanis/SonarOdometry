import numpy as np
import torch
import torch.nn.functional as F
# import pypose as pp
from scipy.spatial.transform import Rotation as R

# === Training metrics - PyTorch === 
def dist_err(x_pred, x_target):
    '''
    Distance betweend two poses, q = (x, y, z).
    Euclidesian norm (L2) from vector difference.
    dist_err = L2(q1 - q2)
    '''
    Lx = torch.linalg.norm(x_target-x_pred, dim=1)
    return Lx
 
def rot_err(q_pred, q_target):
    '''
    Rotation error:
    Angle $\theta$ extracted from difference quaterion $\Delta q$:
    $$\Delta q = q_{pred}^{-1} \otimes q_{target}$$.

    If only rotation angle is needed, this problem can be simoplified to:
    - calculatation of qw quaternion component. 
    - extraction $\theta$ from quaternion construction: w = cos($0.5*\theta$)
    '''
    # find shortest rotation 
    dot = (q_pred * q_target).sum(dim=-1, keepdim=True) 
    q_dist = torch.abs(dot)
    q_dist = torch.clamp(q_dist, max=1.0 - 1e-7)

    return 2*torch.arccos(torch.clamp(q_dist, max = 1 - 1e-7))
    
  
def pose_err(pred, target):

    b, n, _ = pred.shape
    target_act = target[:, :n, :]
    pred = pred.view(b*n, -1)
    target_act = target_act.view(b*n, -1)

    x_pred, x_target = pred[:, :3], target_act[:, :3]
    q_pred, q_target = pred[:, 3:7], target_act[:, 3:7]

    dist = dist_err(x_pred, x_target)
    rot = rot_err(q_pred, q_target)
    
    return torch.mean(dist, dim=-1), torch.mean(rot, dim=-1)



# === EVALUATION METRICS - numpy ===

def align_traj(pred, target, init_pt_only=False):
    trans_pred = pred[:, :3]
    trans_target = target[:, :3]
    q_pred = pred[:, 3:]
    q_target = target[:, 3:]

    # Center trajectory
    if init_pt_only:
        centroid_pred = np.mean(trans_pred[:2, :], axis=0)
        centroid_target = np.mean(trans_target[:2, :], axis=0)
    else:
        centroid_pred = np.mean(trans_pred, axis=0)
        centroid_target = np.mean(trans_target, axis=0)
    
    pred_centered = trans_pred - centroid_pred
    target_centered = trans_target - centroid_target

    # Rotation alignment
    if init_pt_only:
        target_align_base = target_centered[:2, :]
        pred_align_base = pred_centered[:2, :]
    else: 
        target_align_base = target_centered
        pred_align_base = pred_centered

    global_rot, _ = R.align_vectors(target_align_base, pred_align_base)

    # Translation alignment
    global_trans = centroid_target - global_rot.apply(centroid_pred)
    trans_pred_aligned = global_rot.apply(trans_pred) + global_trans

    # Rotation alignment for quaternions
    r = R.from_quat(q_pred)
    r_pred_aligned = global_rot * r
    q_pred_aligned = r_pred_aligned.as_quat()

    pred_aligned = np.concatenate([trans_pred_aligned, q_pred_aligned], axis=1)
    return pred_aligned


def RPE(pred, target, delta=1):
    n = pred.shape[0] - delta
    if n <= 0:
        return np.array([]), 0.0

    Q_i_t = pred[:-delta, :3] 
    Q_i_r = R.from_quat(pred[:-delta, 3:]) 
    Q_d_t = pred[delta:, :3] 
    
    P_i_t = target[:-delta, :3] 
    P_i_r = R.from_quat(target[:-delta, 3:]) 
    P_d_t = target[delta:, :3] 

    # Relative position
    Q_rel_pos = Q_i_r.inv().apply(Q_d_t - Q_i_t)
    P_rel_pos = P_i_r.inv().apply(P_d_t - P_i_t)

    # Translation error norms for each pair
    E = np.linalg.norm(P_rel_pos - Q_rel_pos, axis=1)

    # RMSE over all pairs
    rmse_rpe = np.sqrt(np.mean(E**2))

    return E, rmse_rpe


def ATE(pred, target):
    
    x_pred, x_target = pred[:, :3], target[:, :3]
    q_pred, q_target = pred[:, 3:7], target[:, 3:7]

    # Translation error
    dist = np.linalg.norm(x_target - x_pred, axis=1)

    # Rotation error
    dot = np.sum(q_pred * q_target, axis=1)
    q_dist = np.abs(dot)
    q_dist = np.clip(q_dist, a_min=None, a_max=1.0 - 1e-7) 
    rot = 2 * np.arccos(q_dist)

    # RMSE
    rmse_ate = np.sqrt(np.mean(dist**2))
    rmse_rot = np.sqrt(np.mean(rot**2))

    return rmse_ate, rmse_rot, dist, rot


def eval_metrics(pred, target, align=True, align_init_pt_only=True, add_data_series=False):
  
    n = min(pred.shape[0], target.shape[0])
    pred = pred[:n, :]
    target = target[:n, :]

    if align:
        pred_align = align_traj(pred, target, init_pt_only=align_init_pt_only)
    else:
        pred_align = pred

    # Absolute trajectory error
    rmse_ate, rmse_rot, dist, rot = ATE(pred_align, target)

    # Relative pose error 
    vect_rpe, rmse_rpe = RPE(pred, target)
 
    metrics = {
        # Global metrics
        'RMSE_ATE': float(rmse_ate),
        'RMSE_RPE': float(rmse_rpe),
        'RMSE_ROT': float(rmse_rot),
        
        # Translation stats 
        'MEAN_TRANS_ERR': float(np.mean(dist)),
        'MEDIAN_TRANS_ERR': float(np.median(dist)),
        'STD_TRANS_ERR': float(np.std(dist)),
        'MIN_TRANS_ERR': float(np.min(dist)),
        'MAX_TRANS_ERR': float(np.max(dist)),
        
        # Rotation stats
        'MEAN_ROT_ERR': float(np.mean(rot)),
        'MEDIAN_ROT_ERR': float(np.median(rot)),
        'STD_ROT_ERR': float(np.std(rot)),
        'MIN_ROT_ERR': float(np.min(rot)),
        'MAX_ROT_ERR': float(np.max(rot))

    }
      

    if add_data_series:
        metrics['data_absolute_translation'] = dist
        metrics['data_relative_translation'] = vect_rpe
        metrics['data_absolute_rotation'] = rot
    return metrics

