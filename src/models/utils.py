import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

import pypose as pp


# === Extrinsics calib ===
# T = (sonar_cfg.position.x, sonar_cfg.position.y, sonar_cfg.position.z)
# R = (sonar_cfg.position.roll, sonar_cfg.position.pitch, sonar_cfg.position.yaw)

class ExtrinsicsCalib(nn.Module):
    
    def __init__(self, T, R):
        super().__init__()
        
        T = torch.tensor(T)
        R = torch.tensor(R)
        q = pp.euler2SO3(R).tensor()
        
        r2s = torch.cat([T, q])
        s2r = pp.SE3(r2s).Inv().tensor()
        
        self.register_buffer('se3_r2s', r2s)
        self.register_buffer('se3_s2r', s2r)
        self.register_buffer('z_translation', T[-1])
        
    def pose(self, x, inverse = False):
        se3_x = pp.SE3(x)
        
        if inverse:
            out = se3_x @ pp.SE3(self.se3_s2r)
        else:
            out = se3_x @ pp.SE3(self.se3_r2s)
        return out.tensor()

    def depth(self, x, inverse = False):
        if inverse:
            out = x + self.z_translation
        else:
            out = x - self.z_translation
        return out

    def points(self, x, inverse = False):
        if inverse:
            out = pp.SE3(self.se3_s2r) @ x
        else:
            out = pp.SE3(self.se3_r2s) @ x
        return out
        



# === Points Transformation === 

# carthesian -> polar coord system 
def transform_cart2polar(pts, eps=1e-8):
    x, y, z = pts[:,0], pts[:,1], pts[:,2] 
    
    r_sq_xy = torch.clamp(x**2 + y**2, min=eps)
    r_xy = torch.sqrt(r_sq_xy)
    
    r_sq = r_sq_xy + z**2
    r = torch.sqrt(r_sq)
    
    mask_theta = (torch.abs(x) < eps) & (torch.abs(y) < eps)
    x_s = torch.where(mask_theta, x + eps, x)
    # x_s = torch.where(mask_theta, torch.ones_like(x), x) # alternative version, for points with x = 0 and y = 0, x = 0 -> x = 1 ->  atan2(0.0, 1.0) = 0 
    theta = torch.atan2(y, x_s)

    phi = torch.atan2(z, r_xy)

    return torch.stack((r, theta, phi), dim=1)

# polar -> carthesian coords system
def transform_polar2cart(pts):
    r, theta, phi = pts[:,0], pts[:,1], pts[:,2]
    
    r_xy = r * torch.cos(phi)
    x = r_xy * torch.cos(theta)
    y = r_xy * torch.sin(theta)
    z = r * torch.sin(phi) 
    return  torch.stack((x, y, z), dim=1)
    

# === Project points === 
# Pipe line: 
# 1. Local (source) frame, polar coords sys
# 2. Local (source) frame, carthesian coords sys
# 3. Global frame, carthesian coords sys
# 4. Local (target) frame, carthesian coords sys
# 5. Local (target) frame, polar coords sys

def project_points(origin_pt, origin_pose, target_pose):

    # --- Project origin point from spehrical to cartesian coords sys. (r, theta, phi) -> (x, y, z, 1).T ---
    origin_pt_xyz = transform_polar2cart(origin_pt)

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
    target_pt = target_pt_xyz[:, :3]
    

    # --- Cartesian -> Polar ---
    target_pt = transform_cart2polar(target_pt)

    return target_pt

# === Transform from local to global ===
# transtom points from local, source frame (polar) to global frame (carthesian)
# Pipe line: 
# 1. Local (source) frame, polar coords sys
# 2. Local (source) frame, carthesian coords sys
# 3. Global frame, carthesian coords sys

def transform_to_global(origin_pt, origin_pose):

    n_pts, _ = origin_pt.shape 

    # --- Project origin point from spehrical to cartesian coords sys. (r, theta, phi) -> (x, y, z, 1).T ---
    
    # origin_pt_xyz = transorm_points_coords(origin_pt, projection_type.POLAR2CARTESIAN)
    origin_pt_xyz = transform_polar2cart(origin_pt)

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

# === Movement approximation === 

def approx_movement(x1, x2, t1, t2, t3, motion_model='linear'):
    """
    Approximates the next pose based on the previous two poses using 
    a Constant Velocity Model on the SE(3) Lie Algebra manifold.
    """
    if motion_model == 'linear':
        # Convert raw tensors to PyPose SE3 objects
        pose1 = pp.SE3(x1)
        pose2 = pp.SE3(x2)
        
        # Format time intervals for batch broadcasting.
        # Clamping dt12 prevents division by zero if t1 == t2.
        dt12 = (t2 - t1).view(-1, 1).clamp(min=1e-6)
        dt23 = (t3 - t2).view(-1, 1)

        # Calculate the relative transformation in the local frame
        delta_pose = pose1.Inv() @ pose2 # pose 1 -> pose 2 movement
        
        # Map the SE(3) object to the tangent space se(3) 
        # 7d oboejcts (x, y, z, qx, qy, qz, qw) -> 6d objects (tangent space)
        twist = delta_pose.Log()
        
        # Scale the 6D twist by the time ratio to predict the next movement
        scaled_twist = twist / dt12 * dt23 # now its normal tensor obejct
        
        # Re-cast to a PyPose se3 object and map back to the curved SE(3) manifold
        delta_se3 = pp.se3(scaled_twist).Exp()
        
        # Apply the predicted movement to the last known pose
        pose3 = pose2 @ delta_se3

        return pose3.tensor()
    else:
        # Return last pose 
        return x2
    

def add_noise(poses, noise_intensity = (0.0, 0.0), clear_poses_num=0):
    
    b, n, _ = poses.shape

    noise_6d = (torch.rand((b, n - clear_poses_num, 6), device = poses.device) - 0.5) * 2

    noise_6d[:, :, :3] =  noise_6d[:, :, :3] * noise_intensity[0]
    noise_6d[:, :, 3:] =  noise_6d[:, :, 3:] * noise_intensity[1]

    poses_noise_pp = pp.se3(noise_6d).Exp()
    poses_gt_pp = pp.SE3(poses[:, clear_poses_num:, :])

    poses_gt_noised = poses_gt_pp @ poses_noise_pp

    poses_gt_noised = poses_gt_noised.tensor()
    
    return torch.cat([poses[:, :clear_poses_num, :], poses_gt_noised], dim=1)
    

def depth_to_elev_angle(depth, r):

    depth_r_ratio = torch.clamp(depth/r, -1, 1)
    # Note: Assumption, that there is flat surrounding!
    gt_elevation = torch.asin(depth_r_ratio)  

    return gt_elevation



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


#    if s < 0.001: 
#             q_diff_axis = torch.tensor([1, 0, 0], device = device, dtype = q_diff.dtype)
#         else:

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
