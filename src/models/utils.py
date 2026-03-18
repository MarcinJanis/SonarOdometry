import numpy as np
import torch 
import torch.nn.functional as F

from enum import IntEnum


# === Points Transformation === 
# Transform points between polar and carthesian coords system

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

        # # x = torch.clamp(x, min=1e-5)
        # x = torch.where(x.abs() < 1e-5, torch.sign(x + 1e-9) * 1e-5, x) # safe (no exploding grad) but to not set grad to 0 for points with small x

        # r_sq_xy = torch.clamp(x**2 + y**2, min=1e-8)
        # r_sq = torch.clamp(x**2 + y**2 + z**2, min=1e-8)

        # r = torch.sqrt(r_sq)
        # theta = torch.atan2(y, x)
        # phi = torch.atan2(z, torch.sqrt(r_sq_xy))

        # return torch.stack((r, theta, phi), dim=1)
        
        r_sq_xy = x**2 + y**2
        r_xy = torch.sqrt(r_sq_xy + 1e-8)
        
        r_sq = r_sq_xy + z**2
        r = torch.sqrt(r_sq + 1e-8)

        x_s = torch.where(r_sq_xy < 1e-10, x + 1e-5, x)
        
        theta = torch.atan2(y, x_s)
        phi = torch.atan2(z, r_xy)

        return torch.stack((r, theta, phi), dim=1)
    
    
# === Create Transform Matrix from quaterions=== 
def transform_matrix(state):

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
    row1 = torch.stack([2*(xy + wz), 1 - 2*(xx + zz),  2*(yz - wx), y], dim=1)
    
    # Row 2: [R20, R21, R22, z]
    row2 = torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy),  z], dim=1)
    
    # Row 3: [0, 0, 0, 1] 
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    row3 = torch.stack([zeros, zeros, zeros, ones], dim=1)

    # --- Stack to build transfor matrix ---
    T = torch.stack([row0, row1, row2, row3], dim=1)
    
    return T # shape (N, 4, 4)


# === Project points === 
# Pipe line: 
# 1. Local (source) frame, polar coords sys
# 2. Local (source) frame, carthesian coords sys
# 3. Global frame, carthesian coords sys
# 4. Local (target) frame, carthesian coords sys
# 5. Local (target) frame, polar coords sys

def project_points(origin_pt, origin_pose, target_pose, use_quaterions=True):

    n_pts, _ = origin_pt.shape 

    # --- Project origin point from spehrical to cartesian coords sys. (r, theta, phi) -> (x, y, z, 1).T ---
    origin_pt_xyz = transorm_points_coords(origin_pt, projection_type.POLAR2CARTESIAN)


    if use_quaterions:
        # extract quaterions and translations:
        origin_shift = origin_pose[:, :3]
        origin_rot = origin_pose[:, 3:7]

        target_shift = target_pose[:, :3]
        target_rot = target_pose[:, 3:7]

        # --- origin frame -> global ---
        # rotation:
        global_pt_xyz = hamilton_product(origin_rot, origin_pt_xyz) 
        global_pt_xyz = hamilton_product(global_pt_xyz, q_conjugate(origin_rot))
        # translation: 
        global_pt_xyz = global_pt_xyz[:, :3] + origin_shift

        # --- global -> target frame ---
        # translation:
        target_pt_xyz = global_pt_xyz - target_shift
        # rotation:
        target_pt_xyz = hamilton_product(q_conjugate(target_rot), target_pt_xyz) 
        target_pt_xyz = hamilton_product(target_pt_xyz, target_rot) 
        target_pt_xyz = target_pt_xyz[:, :3]
    else:
        # homogenous coordinates:
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

# === Transform points from local, source frame to global frame, carthesian coords sys
# Pipe line: 
# 1. Local (source) frame, polar coords sys
# 2. Local (source) frame, carthesian coords sys
# 3. Global frame, carthesian coords sys

def transform_to_global(origin_pt, origin_pose):

    n_pts, _ = origin_pt.shape 

    # --- Project origin point from spehrical to cartesian coords sys. (r, theta, phi) -> (x, y, z, 1).T ---
    origin_pt_xyz = transorm_points_coords(origin_pt, projection_type.POLAR2CARTESIAN)
    
    # extract quaterions and translations:
    origin_shift = origin_pose[:, :3]
    origin_rot = origin_pose[:, 3:7]

    # --- origin frame -> global ---
    # rotation:
    global_pt_xyz = hamilton_product(origin_rot, origin_pt_xyz) 
    global_pt_xyz = hamilton_product(global_pt_xyz, q_conjugate(origin_rot))
    # translation: 
    global_pt_xyz = global_pt_xyz[:, :3] + origin_shift

    return global_pt_xyz

# === Pose distance === 

def pose_distance(p1, p2):
    
    # extract translation and rotation
    t1 = p1[:, :3]
    q1 = p1[:, 3:7]

    t2 = p2[:, :3]
    q2 = p2[:, 3:7]

    # linear distance
    dt = t2 - t1
    dist_lin = torch.sqrt(dt[:,0]**2 + dt[:,1]**2 + dt[:,2]**2)

    # # angular distance
    # # q1 * dq = q2 => dq = q1^-1 q2
    # dq = hamilton_product(q_conjugate(q1), q2)
    # dist_ang = #TODO SE(3), log(), norm (?)
    # return dist_lin, dist_ang



# ===    Quaterions algebra   ===
# Note: It is assumed that real coponent of quaterion is on the last position! 
# q = [x, y, z, w]

# if quaterion is unit quaterion (normalized, L2 norm = 1), then:
# quaterion conjugate = transpose quaterion = inverse quaterion 
# (interpretation: same rotation angle but different dircetion)
def q_conjugate(q):
    return q * torch.tensor([-1, -1, -1, 1], device=q.device, dtype=q.dtype)



# Hamiltion product - multiplying two quaterions (real nuber + vector of imaginary components).
# If quaterion passed as an argument has 3 elements, instead of standard 4, it will be assumed that is a point. 
# All 3 values will be assign to imaginary components, and real component will be set to 0. 
def hamilton_product(q1, q2):

    if q1.shape[1] == 3: 
        x1, y1, z1 = q1.unbind(dim=-1)
        w1 = torch.zeros_like(x1) 
    else:
        x1, y1, z1, w1 = q1.unbind(dim=-1)

    if q2.shape[1] == 3: 
        x2, y2, z2 = q2.unbind(dim=-1)
        w2 = torch.zeros_like(x2)
    else:
        x2, y2, z2, w2 = q2.unbind(dim=-1)

    # vector (x, y, z) - complex components
    x =  w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y =  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z =  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    # scalar (w) - real component
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    
    return torch.stack((x, y, z, w), dim=-1)


# # -------------- 
# # Test of points projection. 
# import time 

# n = 100
# points = torch.rand(n, 3)
# origin_pose = torch.rand(n, 7)
# target_pose = torch.rand(n, 7)

# origin_pose[:, 3:7] = F.normalize(origin_pose[:, 3:7], p=2, dim=-1)
# target_pose[:, 3:7] = F.normalize(target_pose[:, 3:7], p=2, dim=-1)

# t0 = time.time()
# output_q = project_points(points, origin_pose,target_pose, use_quaterions=True)
# t1 = time.time()
# output = project_points(points, origin_pose,target_pose, use_quaterions=False)
# t2 = time.time()


# # for i in range(n):
# #     print(f'> pt {i}: {output_q[i, :]} =? {output[i, :]}')

# eps = 1e-6 # permissible tolerance, when 1e-6 -> 100% compliance, 1e-7 -> 87% compliance
# compare_res = torch.sum(torch.sum((torch.abs(output_q - output) <= eps), dim=1) > 0)

# print(f'Test:\ncorrect transforms: {compare_res/n*100} %.')

# print(f'Quaterions exec time: {t1-t0:.4f}\nMatrices exec time: {t2-t1:.4f}')
