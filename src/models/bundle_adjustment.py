import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import transform_cart2polar, transform_polar2cart, depth_to_elev_angle

import pypose as pp



class BundleAdjustment(nn.Module):
    def __init__(self, supervised,
                 init_poses, 
                 gt_poses, gt_depth,
                 init_patch_coords_r_theta, 
                 init_patch_coords_phi, 
                 source_frame_idx, target_frame_idx, patch_idx,
                 delta, weights,
                 sonar_param, freeze_poses):
        
        super().__init__()
        self.device = init_poses.device
        self.supervised = supervised
        # --- init ---
        self.freeze_poses = freeze_poses

        # physical to fls units scaling 
        self.fls2physic_scale_factor = torch.tensor([self.sonar_param.resolution.bins / (self.sonar_param.range.max - self.sonar_param.range.min),
                                            self.sonar_param.resolution.beams / self.sonar_param.fov.horizontal], device = self.device).view(1, 1, 2)


        # remember input shape:
        self.b, _, self.p, _ = init_patch_coords_r_theta.shape
        self.act_n = init_poses.shape[1]
        poses_n = self.b*self.act_n
        self.edges_n = self.b*self.act_n*self.p

        # get actual number of estimated poses 
        init_poses = init_poses.view(1, poses_n, 7)
        init_poses = _quat_norm(init_poses)

        # get acutal number of patch coords
        patch_coords_r_theta = init_patch_coords_r_theta.view(1, self.edges_n, 2)
        patch_coords_phi = init_patch_coords_phi.view(1, self.edges_n, 1)
        

        # --- define parameters to optimize ---
        init_poses_se3 = pp.SE3(init_poses)
        
        if freeze_poses >= self.act_n:
            self.poses_anchor = init_poses_se3
            self.split_poses = False
        elif freeze_poses == 0:
            self.poses_anchor = pp.Parameter(init_poses_se3) # confusing name but dont wnat to make more complex logic which is not necesery
            self.split_poses = False
        else:
            self.poses_anchor = init_poses_se3[:, :freeze_poses, :]
            self.poses_optim = pp.Parameter(init_poses_se3[:, freeze_poses:, :])
            self.split_poses = True
            
        self.elevation_angle = nn.Parameter(patch_coords_phi) 

        # --- define parameters not optimized --- 
        self.patch_coords_r_theta = patch_coords_r_theta
        self.sonar_param = sonar_param


        # global idx -> local idx
        self.source_frame_idx = source_frame_idx % poses_n
        self.target_frame_idx = target_frame_idx % poses_n
        self.patch_idx = patch_idx % self.edges_n

        # --- projection base line ---
        # project points with act pose, add delta
        with torch.no_grad(): # [no grad required !?]
            source_poses = init_poses_se3[:, self.source_frame_idx, :].clone() # [clone() or detach() also!?]
            target_poses = init_poses_se3[:, self.target_frame_idx, :].clone() # [clone() or detach() also!?]
            
            patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :] 
            elevation_angle = self.elevation_angle[:, self.patch_idx].clone() # [clone() or detach() also!?]
            source_coords = torch.cat([patch_coords, elevation_angle], dim = 2)
         
            target_coords = transform(source_poses, target_poses, source_coords)
            
        # projcted coords (r, theta) - baseline for BA optimization
        # here with grad tracking!
        self.coords_baseline = target_coords[:, :, :2] + (delta  / self.fls2physic_scale_factor.squeeze(0))

        # weights for optimization
        self.weights = torch.diag(weights.flatten())
        # [if really needed (NaN occurs), addd to weighs.flatten(), via torch.cat(), small weight, shape: [num_opt_poses+num_opt_elev], with val 0 for elev angl and 1e-4 for poses]
      
        # --- gt data for vicarious loss ---
        if self.supervised:
            gt_poses_cut = gt_poses[:, :self.act_n, :]
            self.act_poses_gt_se3 = pp.SE3(_quat_norm(gt_poses_cut.contiguous().view(1, poses_n, 7)))

            depth_gt_cut = gt_depth[:, :self.act_n]
            depth_gt_expand = depth_gt_cut.contiguous().view(poses_n)[self.source_frame_idx]

            r_coords = patch_coords_r_theta[:, :, 0].contiguous().view(self.edges_n)
            self.act_elev_gt = depth_to_elev_angle(depth_gt_expand, r_coords)

# Powinno być:






    def forward(self, dummy_input=None):

        # compose pose tensor and coord tensor
        if self.split_poses:
            poses = torch.cat([self.poses_anchor, self.poses_optim], dim=1)
        else:
            poses = self.poses_anchor

        patch_coords = self.patch_coords[:, self.patch_idx, :] 
        elevation_angle = self.elevation_angle[:, self.patch_idx, :] # pp.SE3 obj
        source_coords = torch.cat([patch_coords, elevation_angle], dim = 2)


        # expand for all edges
        source_poses = poses[:, self.source_frame_idx, :]
        target_poses = poses[:, self.target_frame_idx, :]

        # --- project --- 
        projected_coords = transform(source_poses, target_poses, source_coords)

        # --- projection error ---
        project_err = (projected_coords[:, :, :2] - self.coords_baseline) * self.fls2physic_scale_factor
        project_err = project_err.view(1, -1)

        # [if needed - add error from:
        # - movement of poses
        # - diffenrece between elev angle]
       
        return project_err


    def run(self, max_iter, early_stop_tol=1e-4, trust_region=2.0):
        
        strategy = pp.optim.strategy.TrustRegion(radius = trust_region)
        # define limits for optimized - damping, radius [m]
        optimizer = pp.optim.LM(self, strategy=strategy)

        init_loss = float('inf')

        # --- optimization loop ---
        with torch.enable_grad():
            for i in range(max_iter):   
                loss = optimizer.step(input = None, weight=self.weights)
        
        # compose optimize opeses and coords
        if self.split_poses:
            pose_optimized_se3 = torch.cat([self.poses_anchor, self.poses_optim], dim=1)
        else:
            pose_optimized_se3 = self.poses_anchor

        pose_optimized = pose_optimized_se3.tensor().detach().view(self.b, self.act_n, 7)
        elevation_optimized = self.elevation_angle.detach().view(self.b, self.act_n, self.p, 1)

        # --- Differentiable Image Correspondences ---
        # Vicarious loss function. 
        # Allow to bypass BA with loss function.
        # Composed optimization iteration don't have to be unrolled whilel grad tractink - no OOM err
        # Avoid problems with .detach() on poses and elev angle
        # Reduce computional gradient graph.

        if self.supervised:
            # in supervised version, network prediction is compared to gt (DPVO)
            ref_poses_detached = self.act_poses_gt_se3.detach()
            ref_elev_detached = self.act_elev_gt.view(1, -1, 1).detach()
        else:
            # in self-supervised version, network prediction is optimization results (iMatching)
            ref_poses_detached = pose_optimized_se3.detach()
            ref_elev_detached = self.elevation_angle[:, self.patch_idx, :].view(1, -1, 1).detach()

        source_poses = ref_poses_detached[:, self.source_frame_idx, :]
        target_poses = ref_poses_detached[:, self.target_frame_idx, :]

        patch_coords = self.patch_coords_r_theta[:, self.patch_idx, :] 
        source_coords = torch.cat([patch_coords, ref_elev_detached], dim=2)

        with torch.no_grad():
            projected_coords = transform(source_poses, target_poses, source_coords)

        predicted_projection = self.coords_baseline * self.fls2physic_scale_factor
        target_projection = projected_coords[:, :, :2] * self.fls2physic_scale_factor
    
        return pose_optimized, elevation_optimized, predicted_projection, target_projection



def transform(source_poses, target_poses, coords):
    # project points from source pose to target pose frame of refernce
    # function operates on pp.SE3() objects

    source_poses = source_poses.squeeze(0)
    target_poses = target_poses.squeeze(0)
    coords = coords.squeeze(0)

    local_source_coords = transform_polar2cart(coords) 

    global_coords = source_poses @ local_source_coords

    local_target_coords = target_poses.Inv() @ global_coords

    coords = transform_cart2polar(local_target_coords)

    return coords.unsqueeze(0)

def _quat_norm(pose):
    pose[:, :, 3:] = F.normalize(pose[:, :, 3:], p=2, dim=-1)
    return pose
       
