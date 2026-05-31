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

