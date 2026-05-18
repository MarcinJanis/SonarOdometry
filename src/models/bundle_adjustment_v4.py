import torch
import torch.nn as nn
import pypose as pp
from .utils import project_points 

class BundleAdjustment(nn.Module):
    def __init__(self, 
                 init_poses, 
                 init_patch_coords_r_theta, init_patch_coords_phi, 
                 source_frame_idx, target_frame_idx, patch_idx,
                 delta, weights, freeze_poses, 
                 model_config, sonar_config, 
                 damping=False):

      super().__init__()
      self.device = init_poses.device
        
      self.freeze_poses = max(1, freeze_poses)
        
      self.r_min = sonar_config.range.min
      self.r_max = sonar_config.range.max
      self.fov_horizontal = sonar_config.fov.horizontal
      self.fls_h = model_config.FLS_INPUT_HEIGHT
      self.fls_w = model_config.FLS_INPUT_WIDTH

      self.b, self.n_total, self.p, _ = init_patch_coords_r_theta.shape
      self.act_n = init_poses.shape[1]
      poses_n = self.b * self.act_n
        
      self.s_idx = (source_frame_idx % poses_n).squeeze() # source frame local index
      self.t_idx = (target_frame_idx % poses_n).squeeze() # target frame local index 
      self.patch_idx = (patch_idx % (self.b * self.act_n * self.p)).squeeze() # patch local idx

      # poses
      self.init_poses_se3 = pp.SE3(init_poses)
      poses_correction = torch.zeros((self.b, self.n_total - self.freeze_poses, 7), device = self.device)   
      poses_correction[:, :, -1] = torch.ones((self.b, self.n_total - self.freeze_poses), device = self.device)            
      self.poses_correction = pp.Parameters(pp.SE3(poses_correction))

      # patch coords
      self.patch_coords_r_theta = init_patch_coords_r_theta
      self.patch_coords_phi = nn.Parameter(init_patch_coords_phi)              
                     
  def _get_opt_poses(self):
      freezed_poses = torch.zeros((self.b, self.freeze_poses, 7), device = self.device)
      freezed_poses[:, :, -1] = torch.ones((self.b, self.freeze_poses), device = self.device)  
      poses_correction_full = torch.cat((freezed_poses, self.poses_correction.detach()), dim=1)
      return self.init_poses_se3 @ poses_correction_full
      
  def _baseline(self, poses, delta):

      patch_coords = torch.cat((self.patch_coords_r_theta, self.patch_coords_phi), dim=-1)
      
      source_poses = poses[self.s_idx]
      target_poses = poses[self.t_idx]

       = project_points()
      












    
