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
        r_xy = r * np.cos(phi)
        x = r_xy * np.cos(theta)
        y = r_xy * np.sin(theta)
        z = r * np.sin(phi) 
        return  np.column_stack((x, y, z))
    
    elif mode == projection_type.CARTESIAN2POLAR:
        x, y, z = pts[:,0], pts[:,1], pts[:,2] 

        # phi = np.arctan2(z, np.sqrt(x**2 + y**2))
        # r = z / np.sin(phi)
        # theta = np.arctan2(y, z)

        r = np.sqrt(x**2 +  y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(z, np.sqrt(x**2 + y**2))

        return np.column_stack((r, theta, phi))
        
def transform_matrix(state):
    x, y, z, qx, qy, qz, qw = state # quaterions 

    # pre-calculation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    # compose translation matrix
    row0 = torch.stack([1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy),    x])
    row1 = torch.stack([    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx),    y])
    row2 = torch.stack([    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy),    z])
    row3 = torch.tensor([           0.0,             0.0,             0.0,  1.0], device=state.device, dtype=state.dtype)
 
    T = torch.stack([row0, row1, row2, row3])
    return T

def inverse_transform_matrix(T):

    # extract rotation and shift 
    R = T[:3, :3]
    t = T[:3, 3]

    # inverse components
    R_inv = R.T
    t_inv = - torch.mv(R_inv, t)

    # compose inverse matrix 
    T_inv = torch.zeros((4,4), device = T.device, dtype = T.dtype)
    
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    T_inv[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device = T.device, dtype = T.dtype)
    return T_inv 
    
def project_points(origin_pt, origin_pose, target_pose):

    '''
    Calculate projection of point from origin patch to target patch. 
    Note: origin_pt shall be already scaled to real-world values, not in pixels. 
    '''
    # --- Project origin point from spehrical to cartesian coords sys. (r, theta, phi) -> (x, y, z) ---
    origin_pt_xyz = transorm_points_coords(origin_pt, projection_type.POLAR2CARTESIAN)
    origin_pt_xyz = origin_pt_xyz.T 

    # --- Compose relative translation matrix from origin pose to target pose --- 
    T_origin = transform_matrix(origin_pose)
    T_target = transform_matrix(target_pose)
    T_target_inv = inverse_transform_matrix(T_target)
    T_relative = T_target_inv @ T_target

    # --- Calculate translated point coords: Origin pose (x1, y1, z1) -> Target pose (x2, y2, z2) ---
    target_pt_xyz = T_relative * origin_pt_xyz
    target_pt_xyz = target_pt_xyz.T

    # --- Project target point to spherical coords system  ---
    target_pt = transorm_points_coords(target_pt_xyz, projection_type.CARTESIAN2POLAR)
    
    return target_pt


# =======================
# === Quaterions math ===
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
