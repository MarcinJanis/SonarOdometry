import numpy as np
import torch
import pypose as pp

from scipy.spatial.transform import Rotation as R

def align_traj(pred, target, init_pt_only = False):

    trans_pred = pred[:, :3]
    trans_target = target[:, :3]

    q_pred = pred[:, 3:]
    q_target = target[:, 3:]
    # center trajectory
    if init_pt_only:
        centroid_pred = np.mean(trans_pred[:2, :], axis=0)
        centroid_target = np.mean(trans_target[:2, :], axis=0)
    else:
        centroid_pred = np.mean(trans_pred, axis=0)
        centroid_target = np.mean(trans_target, axis=0)
    
    pred_centered = trans_pred - centroid_pred
    target_centered = trans_target - centroid_target

    # rotation pred -> aligned pred
    if init_pt_only:
        target_align_base = target_centered[:2, :]
        pred_align_base = pred_centered[:2, :]
    else: 
        target_align_base = target_centered
        pred_align_base = pred_centered

    global_rot, _ = R.align_vectors(target_align_base, pred_align_base)

    # translation pred -> aligned pred
    # t1_2 = Rt1 + t2
    global_trans = centroid_target - global_rot.apply(centroid_pred)
    
    # apply transformations for translation
    trans_pred_aligned = global_rot.apply(trans_pred) + global_trans

    # apply transformations for rotation
    r = R.from_quat(q_pred)
    r_pred_aligned = global_rot * r
    q_pred_aligned = r_pred_aligned.as_quat()

    pred_aligned = np.concatenate([trans_pred_aligned, q_pred_aligned], axis=1)
    return pred_aligned

import torch
import torch.nn.functional as F








def dist_err(x_pred, x_target):
    '''
    Euclidesian norm (L2) from vector difference.
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
    # 
    # find shortest rotation 
    dot = (q_pred * q_target).sum(dim=-1, keepdim=True) 
    q_dist = torch.abs(dot)
    q_dist = torch.clamp(q_dist, max=1.0 - 1e-7)

    return 2*torch.arccos(torch.clamp(q_dist, max = 1 - 1e-7))
    
  
def ATE_mean(pred, target):

    b, n, _ = pred.shape
    target_act = target[:, :n, :]
    pred = pred.view(b*n, -1)
    target_act = target_act.view(b*n, -1)

    x_pred, x_target = pred[:, :3], target_act[:, :3]
    q_pred, q_target = pred[:, 3:7], target_act[:, 3:7]

    dist = dist_err(x_pred, x_target)
    rot = rot_err(q_pred, q_target)
    
    return torch.mean(dist, dim=-1), torch.mean(rot, dim=-1)

    

def eval_metics(pred, target):

    n_pred = pred.shape[0]
    n_target = pred.shape[0]

    n = min(n_pred, n_target)

    x_pred, x_target = pred[:n, :3], target[:n, :3]
    q_pred, q_target = pred[:n, 3:7], target[:n, 3:7]

    dist = dist_err(x_pred, x_target)
    rot = rot_err(q_pred, q_target)
    
    rsme_ate = torch.sqrt(torch.sum(dist**2) / n)
    rsme_rot_err = torch.sqrt(torch.sum(rot**2) / n)

    mean_ate = torch.mean(dist, dim=-1)
    mean_rot = torch.mean(rot, dim=-1)
    #TODO: RPE
    metrics = {
        'RSME_ATE': rsme_ate,
        'RSME_ROT_ERR':

    } 

    return metrics
   

