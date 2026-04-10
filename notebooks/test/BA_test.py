import torch 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from src.models.utils import project_points

def generate_data(n_poses, n_pts):
  
  pose_val_max = 20.0
  pose_noise_max = 1.0

  r_min = 2.0
  r_max = 100.0

  theta_min 
  theta_max 

  phi_min
  phi_max
  
  # gt poses
  poses_gt = torch.rand((1, n_poses, 7)) * pose_val_max
  poses_gt[:, :, 3:] = F.normalize(poses_gt[:, :, 3:], p=2.0, dim=-1)

  # noised poses  
  poses_noise = poses_gt +  torch.rand((1, n_poses, 7)) * pose_noise_max
  poses_noise[:, :, 3:] = F.normalize(poses_noise[:, :, 3:], p=2.0, dim=-1)

  # points coords
  r = torch.rand(1, n_poses, n_pts, 1) * (r_max - r_min) + r_min
  theta = torch.rand(1, n_poses, n_pts, 1) * (theta_max - theta_min) + theta_min
  phi = torch.rand(1, n_poses, n_pts, 1) * (phi_max - phi_min) + phi_min

  coords_r_theta = torch.cat([r, theta], dim=-1)
  coords_stack = torch.cat([r, theta, phi], dim=-1)
  
  # edges 
  new_i, new_j = [], [] 
  for sf in range(n_poses): # for each source frame
    for tf in range(n_poses): # for each target frame
        if sf != tf: 
          # edges: new patches -> old frames
          new_i.append(torch.arange(sf * self.patches_per_frame, (sf + 1) * self.patches_per_frame, device=device)) 
          new_j.append(torch.full((self.patches_per_frame,), tf, device=device))
          
          # edges: old patches -> new frame
          new_i.append(torch.arange((tf) * self.patches_per_frame, (tf + 1) * self.patches_per_frame, device=device))
          new_j.append(torch.full((self.patches_per_frame,), sf, device=device))

  i = torch.cat(new_i, dim=0)
  j = torch.cat(new_j, dim=0)
  edges_num = i.shape[0]

  source_poses_idx = i // n_pts
  patch_idx = i
  target_poses_idx = j 
  
  source_coords = coords_stack.view(n_poses*n_pts, 3)[patch_idx]
  
  source_poses_gt = poses_gt.view(n_poses, 7)[source_poses_idx]
  target_poses_gt = poses_gt.view(n_poses, 7)[target_poses_idx]
  
  projection_gt = project_points(source_coords, origin_poses, target_poses)   

  source_poses_noise = poses_noise.view(n_poses, 7)[source_poses_idx]
  target_poses_noise = poses_noise.view(n_poses, 7)[target_poses_idx]
  
  projection_noise = project_points(source_coords, origin_poses, target_poses)  
            
  delta = (projection_gt - projection_noise)[:, :2]
  
  return 
  
  

