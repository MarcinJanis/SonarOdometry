import numpy as np
import torch 

from enum import IntEnum

# =============================
# === Points tranformations ===
# =============================

class projection_type(IntEnum):
    POLAR2CARTESIAN = 0
    CARTESIAN2POLAR = 1

def transorm_points_coords(pts, mode:projection_type):
    '''

    :pt: numpy array with points
    if projection_type == POLAR2CARTESIAN:
        pt[:,0] -> r
        pt[:,1] -> theta
        pt[:,2] -> phi
    
    if projection_type == CARTESIAN2POLAR:
        pt[:,0] -> x
        pt[:,1] -> y
        pt[:,2] -> z

    :mode: projection_type
    '''

    if mode == projection_type.POLAR2CARTESIAN:
        r, theta, phi = pts[:,0], pts[:,1], pts[:,2]
        r_xy = r * torch.cos(phi)
        x = r_xy * torch.cos(theta)
        y = r_xy * torch.sin(theta)
        z = r * torch.sin(phi) 
        return  torch.stack((x, y, z), dim=1)
    
    elif mode == projection_type.CARTESIAN2POLAR:
        x, y, z = pts[:,0], pts[:,1], pts[:,2] 

        # phi = np.arctan2(z, np.sqrt(x**2 + y**2))
        # r = z / np.sin(phi)
        # theta = np.arctan2(y, z)
        r_sq_xy = torch.clamp(x**2 + y**2, min=1e-8)
        r_sq = torch.clamp(x**2 + y**2 + z**2, min=1e-8)

        r = torch.sqrt(r_sq)
        theta = torch.atan2(y, x + 1e-8)
        phi = torch.atan2(z, torch.sqrt(r_sq_xy))

        return torch.stack((r, theta, phi), dim=1)
        
# def transform_matrix(state):
#     x, y, z, qx, qy, qz, qw = state # quaterions 

#     # pre-calculation
#     xx = qx * qx
#     yy = qy * qy
#     zz = qz * qz
#     xy = qx * qy
#     xz = qx * qz
#     yz = qy * qz
#     wx = qw * qx
#     wy = qw * qy
#     wz = qw * qz

#     # compose translation matrix
#     row0 = torch.stack([1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy),    x])
#     row1 = torch.stack([    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx),    y])
#     row2 = torch.stack([    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy),    z])
#     row3 = torch.tensor([           0.0,             0.0,             0.0,  1.0], device=state.device, dtype=state.dtype)
 
#     T = torch.stack([row0, row1, row2, row3])
#     return T

def transform_matrix(state):
    '''
    Converts poses to transformation matrices (batch processing).
    
    :state: tensor (N, 7) [x, y, z, qx, qy, qz, qw]
    :return: tensor (N, 4, 4)
    '''
    
    # -- -Extrac shift and rotation components for each pose ---
    x, y, z = state[:, 0], state[:, 1], state[:, 2]
    qx, qy, qz, qw = state[:, 3], state[:, 4], state[:, 5], state[:, 6]

    # --- Pre-calculation --- 
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    # Construct rows 
    
    # Row 0: [R00, R01, R02, x]
    row0 = torch.stack([1 - 2*(yy + zz),  2*(xy - wz),      2*(xz + wy),      x], dim=1)
    
    # Row 1: [R10, R11, R12, y]
    row1 = torch.stack([2*(xy + wz),      1 - 2*(xx + zz),  2*(yz - wx),      y], dim=1)
    
    # Row 2: [R20, R21, R22, z]
    row2 = torch.stack([2*(xz - wy),      2*(yz + wx),      1 - 2*(xx + yy),  z], dim=1)
    
    # Row 3: [0, 0, 0, 1] 
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    row3 = torch.stack([zeros, zeros, zeros, ones], dim=1)

    # --- Stack to build transfor matrix ---
    T = torch.stack([row0, row1, row2, row3], dim=1)
    
    return T # shape (N, 4, 4)


# def inverse_transform_matrix(T):

#     # extract rotation and shift 
#     R = T[:3, :3]
#     t = T[:3, 3]

#     # inverse components
#     R_inv = R.T
#     t_inv = - torch.mv(R_inv, t)

#     # compose inverse matrix 
#     T_inv = torch.zeros((4,4), device = T.device, dtype = T.dtype)
    
#     T_inv[:3, :3] = R_inv
#     T_inv[:3, 3] = t_inv
#     T_inv[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device = T.device, dtype = T.dtype)
#     return T_inv 
    
def project_points(origin_pt, origin_pose, target_pose):

    '''
    Calculate projection of point from origin patch to target patch. 
    Note: origin_pt shall be already scaled to real-world values, not in pixels. 
    '''
    n_pts, _ = origin_pt.shape 

    # --- Project origin point from spehrical to cartesian coords sys. (r, theta, phi) -> (x, y, z, 1).T ---
    origin_pt_xyz = transorm_points_coords(origin_pt, projection_type.POLAR2CARTESIAN)

    ones = torch.ones((n_pts, 1), device=origin_pt.device, dtype=origin_pt.dtype)
    origin_pt = torch.cat((origin_pt_xyz, ones), dim=1).unsqueeze(-1) # Shape: (N, 4, 1)

    # --- Transform matrixes ---
    T_origin = transform_matrix(origin_pose)       # (N, 4, 4)
    T_target = transform_matrix(target_pose)       # (N, 4, 4)
    
    # Inverse target transform matrix
    T_target_inv = torch.linalg.inv(T_target)      # (N, 4, 4)

    # Relative transform matrxi 
    T_relative = T_target_inv @ T_origin

    # --- transform ---
    target_pt= T_relative @ origin_pt

    target_pt_xyz = target_pt[:, :3, 0] 

    # --- Cartesian -> Polar ---
    target_pt = transorm_points_coords(target_pt_xyz, projection_type.CARTESIAN2POLAR)

    return target_pt


# =======================
# ===    Quaterions   ===
# =======================

def q_conjugate(q):
    # quaterion conjugate
    return q * torch.tensor([-1, -1, -1, 1])


def hamilton_product(q1, q2):
    '''
    Calculate Hamiltion product for two quaterion tensors.
    q = [x, y, z, w] -> w + i*x + j*y + k*z
    '''
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)

    # vector (x, y, z)
    x =  w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y =  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z =  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    # scalar (w)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    
    return torch.stack((x, y, z, w), dim=-1)
