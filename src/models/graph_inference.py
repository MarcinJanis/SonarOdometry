import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import os 
import csv

import numpy as np 

from .patchifier import Patchifier
from .utils import hamilton_product, q_conjugate, project_points


class Graph(nn.Module):
  def __init__(self, model_cfg, sonar_cfg, output_dir):
    super().__init__()

    # --- import sonar configuration ---
    self.r_min = sonar_cfg.range.min # min range
    self.r_max = sonar_cfg.range.max # max range

    self.fls_h = sonar_cfg.resolution.bins # vertical resolution of input fls image
    self.fls_w = sonar_cfg.resolution.beams # horizontal resolution of input fls image

    self.fov_vertical = sonar_cfg.fov.vertical  * math.pi / 180 # vertical fov [deg]
    self.fov_horizontal = sonar_cfg.fov.horizontal  * math.pi / 180 # horizontal fov [deg]

    # --- import sys configuration ---
    self.buff_size = model_cfg.BUFF_SIZE # ring buffer size
  
    self.patches_per_frame = model_cfg.PATCHES_PER_FRAME # amount of patches generated per each frames
    self.patch_size = model_cfg.PATCH_SIZE # size of each patch, patch shape: (c, p, p)

    self.time_window = model_cfg.TIME_WINDOW # time window in frames history in which patches are tracked
    
    self.fmap_c = model_cfg.FEATURES_OUTPUT_CH # channels num of encoder output 
    self.cmap_c = model_cfg.CONTEXT_OUTPUT_CH # context features = hidden state features 

    self.corr_neighbour = model_cfg.CORR_NEIGHBOUR # size of nieghbour of projected patch that is used in correlation calculations
    self.fmap_h = sonar_cfg.resolution.bins // model_cfg.ENCODER_DOWNSIZE # feature map size h
    self.fmap_w = sonar_cfg.resolution.beams // model_cfg.ENCODER_DOWNSIZE # feature map size w
    self.encoder_downsize = model_cfg.ENCODER_DOWNSIZE # encoder downsize factor 
    
    self.motion_model = model_cfg.MOTION_APPRO_MODEL # method used in initial estimating of new pose
    self.patchifier_method = model_cfg.PATCHIFIER_METHOD # method used in key points detection
    self.grid_size = (model_cfg.PATCHES_GRID_SIZE.y, model_cfg.PATCHES_GRID_SIZE.x) # grid size used to detect key points from whole image equally

    # --- Patchifier ---
    self.patchifier = Patchifier(model_cfg, debug_mode = False)
    
    # === Graph initialization ===
    self.frame_n = 0 # frame counter for ring buffer 

    # --- poses and time stamp buffers ---
    self.register_buffer('time', torch.zeros((self.buff_size), dtype=torch.float)) # time stamp
    self.register_buffer('poses', torch.zeros((self.buff_size, 7), dtype=torch.float)) # poses 

    # --- frame buffers ---
    self.register_buffer('fmap1', torch.zeros((self.buff_size, self.fmap_c, self.fmap_h, self.fmap_w), dtype = torch.float)) # frames: features map
    self.register_buffer('fmap2', torch.zeros((self.buff_size, self.fmap_c, self.fmap_h // self.encoder_downsize, self.fmap_w // self.encoder_downsize), dtype = torch.float)) # frames: features map 

    # --- patches buffers ---
    self.register_buffer('patches_f', torch.zeros((self.buff_size, self.patches_per_frame,  self.fmap_c, self.patch_size*self.patch_size), dtype = torch.float)) # patches: features
    self.register_buffer('patches_c', torch.zeros((self.buff_size, self.patches_per_frame,  self.cmap_c), dtype = torch.float)) # patches: context

    # --- patch center coords buffer ---
    self.register_buffer('patch_state', torch.zeros((self.buff_size, self.patches_per_frame, 3), dtype = torch.float)) # points (r, theta, phi) refered to patches in real world units
    self.register_buffer('hidden_state', torch.zeros((self.buff_size, self.patches_per_frame, self.cmap_c), dtype = torch.float))

    # --- source frame buffer ---
    self.register_buffer('source_frame', torch.zeros((self.buff_size, self.patches_per_frame), dtype = torch.int)) # id of source frame for each patch
                         
    # --- graphs edges --- 
    self.max_edges = self.buff_size * self.patches_per_frame * self.time_window * 2 # max amount of edges in graph 

    self.register_buffer('i', torch.full((self.max_edges,), -1, dtype=torch.long)) # keeps idxs of patch
    self.register_buffer('j', torch.full((self.max_edges,), -1, dtype=torch.long)) # keeps idxs of frame
    # self.register_buffer('weights', torch.zeros(self.max_edges, dtype=torch.float)) # weights of each patch, how good estimation is, based on this patch 
  

    # --- init output files --- 

    if not output_dir is None: 

      os.makedirs(output_dir, exist_ok=True)
      self.output_dir = output_dir

      self.outputf_init(output_dir)
      
  def __del__(self):
    try:
        self.outputf_close()
    except:
        pass
    
    
  # --- data acquisition fcn ---

  def outputf_init(self, output_dir):
    # primary_traj_file_pth = os.path.join(self.output_dir, "trajectory_primary_estimation.csv")
    secondary_traj_file_pth = os.path.join(self.output_dir, "trajectory_secondary_estimation.csv")
    points3d_file_pth = os.path.join(self.output_dir, "3d_points_estimation.csv")

    # create references to file
    # self.f_prim_traj = open(primary_traj_file_pth, mode='a', newline='')
    # self.w_prim_traj = csv.writer(self.f_prim_traj)
    
    self.f_sec_traj = open(secondary_traj_file_pth, mode='a', newline='')
    self.w_sec_traj = csv.writer(self.f_sec_traj)

    self.f_pts3d = open(points3d_file_pth, mode='a', newline='')
    self.w_pts3d = csv.writer(self.f_pts3d)
    
    # define headers
    # if self.f_prim_traj.tell() == 0: 
    #   self.w_prim_traj.writerow(['pose_no', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw' ])

    if self.f_sec_traj.tell() == 0: 
      self.w_sec_traj.writerow(['pose_no', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw' ])

    if self.f_pts3d.tell() == 0: 
      self.w_pts3d.writerow(['n', 'x', 'y', 'z'])


  def outputf_write_pose(self, num, time, sec_traj):
      data_sec_traj = sec_traj.detach().cpu().numpy()
      data_sec_traj = np.concatenate([np.expand_dims(num, axis=0), np.expand_dims(time, axis=0), data_sec_traj], axis = 0)
      self.w_sec_traj.writerow(data_sec_traj)

  def outputf_write_pts(self, num1, num2, pts):
      data_pts = pts.detach().cpu().numpy()
      num = np.expand_dims(np.arange(num1, num2), axis=1)
      data_pts = np.concatenate([num, data_pts], axis = 1)
      self.w_pts3d.writerow(data_pts)

  def outputf_close(self):
    self.f_sec_traj.close()
    self.f_pts3d.close()

# --- Append Graph fcn --- 

  def add_frame(self, fmap, time_stamp, device): 
    # local index for ring buffer 
    local_idx = self.frame_n % self.buff_size

    # Assign to buffers. Create pyramid of features map
    self.fmap1[local_idx, :, :, :] = F.avg_pool2d(fmap.squeeze(0), 1, 1) 
    self.fmap2[local_idx, :, :, :] = F.avg_pool2d(fmap.squeeze(0), self.encoder_downsize, self.encoder_downsize)
    self.time[local_idx] = time_stamp 
    return 
      
  def _scale_fls2phisical(self, coords):
  
    # range r - measured by sonar
    r_norm = coords[:, 1] / self.fls_h
    r = r_norm * (self.r_max - self.r_min) + self.r_min

    # azimuth angle theta - measured by sonar
    theta_norm = coords[:, 0] / self.fls_w - 0.5
    theta = theta_norm * self.fov_horizontal * torch.pi / 180.0
    
    return torch.stack([r, theta], dim = 1)

  def _scale_phisical2fls(self, coords):

    # range r - measured by sonar
    r_norm = (coords[:, 0] - self.r_min) / (self.r_max - self.r_min)
    r = r_norm * self.fls_h

    # azimuth angle theta - measured by sonar
    theta_norm = coords[:, 1] * 180.0 / torch.pi / self.fov_horizontal 
    theta = (theta_norm + 0.5) * self.fls_w
    
    return torch.stack([r, theta], dim = 1)
  
  def add_patches(self, patches_f, patches_c, coords, device):

    # local index for ring buffer
    local_idx = self.frame_n % self.buff_size

    # add patches features to graph 
    self.patches_f[local_idx, :, :, :] = patches_f.squeeze(0) 
    self.patches_c[local_idx, :, :] = patches_c.squeeze(0) 

    # rescale coords to real world values  
    phisical_coords = self._scale_fls2phisical(coords.squeeze(0).squeeze(0))
    
    r, theta = phisical_coords[:, 0], phisical_coords[:, 1]

    # approximate elevation angle - phi - to be optimized 
    phi = torch.zeros((self.patches_per_frame), device = device, dtype = torch.float)
    
    # save optimized pts
    optimize_pts3d = self.patch_state[local_idx, :, :]
    
    self.outputf_write_pts((self.frame_n - 1) * self.patches_per_frame, self.frame_n * self.patches_per_frame, optimize_pts3d)

    # add new to graph 
    self.patch_state[local_idx, :, :] = torch.stack([r.squeeze(0), theta.squeeze(0), phi], dim=1)
    
    # add source frame id to graph
    self.source_frame[local_idx, :] = self.frame_n 
    return 

  def approx_movement(self, device):
    
    k_idx = self.frame_n % self.buff_size

    # print(f'act n: {self.frame_n}')
    if self.frame_n < 2: # initialization
      x0 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float)
    else: 
      # --- indexes ---
      k1_idx = (k_idx - 1) % self.buff_size  
      k2_idx = (k_idx - 2) % self.buff_size

      # --- get time stams ---
      t0 = self.time[k_idx]
      t1 = self.time[k1_idx]
      t2 = self.time[k2_idx]
      
      assert t0 != t1, f'[Error] Time stamps for frame {self.frame_n} and {self.frame_n - 1} are the same.\nMovement approximation is not possible.\n time vector: {self.time}'
      assert t1 != t2, f'[Error] Time stamps for frame {self.frame_n - 1} and {self.frame_n - 2} are the same.\nMovement approximation is not possible.\n time vector: {self.time}'
      
      # --- get previous position ---
      x1 = self.poses[k1_idx, :]
      x2 = self.poses[k2_idx, :]

      if self.motion_model == 'linear':

        # --- linear displacement ---
        new_pose = x1[0:3] + (x1[0:3] - x2[0:3])/(t1 - t2)*(t0 - t1) 

        # --- extract quaterions ---
        q1 = x1[3:]
        q2 = x2[3:]

        # --- find shortest rotation ---
        dot = (q1 * q2).sum() 
        if dot < 0: q1 = -q1

        # --- rotation - quaterions difference ---
        q2_conj = q_conjugate(q2)
        diff = hamilton_product(q1, q2_conj)  # diff q2 -> q1: diff = q2 * q1^-1

        # --- rotation axis ---
        s = torch.sqrt(torch.clamp(1 - diff[-1]*diff[-1], 0.0))
        if s < 0.001: rot_axis = torch.tensor([1, 0, 0], device = device, dtype = diff.dtype)
        else: rot_axis = diff[:-1]/s

        # --- rotation angle ---
        rot_angle = 2 * torch.acos(torch.clamp(diff[-1], -1.0, 1.0))
        rot_angle_appro = rot_angle/(t1 - t2)*(t0 - t1) # approximation apriori, to t0

        # --- apply rotation ---
        q_step_vect = rot_axis * torch.sin(rot_angle_appro/2)
        q_step_scal = torch.cos(rot_angle_appro/2).unsqueeze(0)   
        q_step = torch.cat((q_step_vect, q_step_scal), dim=0)

        q0 = hamilton_product(q_step, q1)
        q0 = q0 /torch.norm(q0)

        x0 = torch.cat((new_pose, q0), dim=0)

      else:
        x0 = x1

    # --- save to buffer ---

    # get old optimize pose
    if self.frame_n > self.buff_size:
      optimize_pose = self.poses[k_idx, :]
      self.outputf_write_pose(self.frame_n - self.buff_size, self.time[k_idx], optimize_pose)

    # overwrite new 
    self.poses[k_idx, :] = x0

    return 


# --- Create connections in Graph --- 

  def create_edges(self, device):    
    '''
    Create set of edges in graph based on new frame and new patches 
    '''

    # --- current patches -> past frame --- 
    new_patches = torch.arange(self.frame_n*self.patches_per_frame, (self.frame_n+1)*self.patches_per_frame, device = device, dtype = torch.long) 
    past_frames = torch.arange(self.frame_n - 1, self.frame_n - 1- self.time_window, step=-1, device = device, dtype = torch.long)
    # past_frames = torch.clamp(past_frames, min=0)

    i_new_patches = new_patches.repeat(self.time_window)
    j_past_frames = torch.repeat_interleave(past_frames, repeats=self.patches_per_frame)
    
    # --- past patches -> current frame --- 
    i_past_patches = torch.arange((self.frame_n - self.time_window)*self.patches_per_frame, self.frame_n*self.patches_per_frame, device = device, dtype = torch.long)
    j_new_frames = torch.ones(self.time_window*self.patches_per_frame, device = device, dtype = torch.long) * self.frame_n 

    # --- concat --- 
    new_i = torch.cat((i_new_patches, i_past_patches), dim = 0)
    new_j = torch.cat((j_past_frames, j_new_frames), dim = 0)

    chunk_size = self.patches_per_frame * self.time_window * 2
    idx_low = (self.frame_n % self.buff_size) * chunk_size
    idx_high = idx_low + chunk_size

    self.i[idx_low:idx_high] = new_i
    self.j[idx_low:idx_high] = new_j

    # self.weights[idx_low:idx_high] = torch.zeros((idx_high - idx_low), device = device, dtype = torch.long)
    # self.valid[idx_low:idx_high] = torch.ones((idx_high - idx_low))

    return 
  
  def corr(self, device):
    '''
    Calculate correlation of patches with their actual fitting
    '''
    
    # --- get source poses and target poses --- 
    # each new frame creates 2*time_window*patches_per_frame new edges, max edges is 2*time_window*patches_per_frame new edges*buff_size
    if self.frame_n < self.buff_size:
      i = self.i[:(self.frame_n+1)*2*self.time_window*self.patches_per_frame]
      j = self.j[:(self.frame_n+1)*2*self.time_window*self.patches_per_frame]
    else:
      i = self.i
      j = self.j

    source_frames_idx = i // self.patches_per_frame
    local_patch_idx = i % self.patches_per_frame

    buff_source_frame_idx = source_frames_idx % self.buff_size
    buff_target_frame_idx = j % self.buff_size

    source_poses = self.poses[buff_source_frame_idx, :]
    target_poses = self.poses[buff_target_frame_idx, :]

    source_coords = self.patch_state[buff_source_frame_idx, local_patch_idx, :]
    
    # --- reprojection ---
    target_pts = project_points(source_coords, source_poses, target_poses)
  
    # --- edges validation --- 

    # find non valid edges
    theta_max = self.fov_horizontal / 2
    phi_max = self.fov_vertical / 2

    out_of_range = (target_pts[:,0] < self.r_min) | (target_pts[:,0] > self.r_max)
    out_of_range = out_of_range | (torch.abs(target_pts[:,1]) > theta_max)
    out_of_range = out_of_range | (torch.abs(target_pts[:,2]) > phi_max)

    out_of_range = out_of_range | (i < 0)
    out_of_range = out_of_range | (j < 0)

    # discard non valid edges
    valid_mask = ~out_of_range
    target_pts = target_pts[valid_mask]
    buff_source_frame_idx = buff_source_frame_idx[valid_mask]
    buff_target_frame_idx = buff_target_frame_idx[valid_mask]
    local_patch_idx = local_patch_idx[valid_mask]
    
    pts_num = target_pts.shape[0]
    
    # print(f'total edges: {source_coords.shape[0]}, valid edges: {torch.sum(valid_mask.long())}')
    # --- get correlation neighbour from fmap --- 
    target_pts_fls = self._scale_phisical2fls(target_pts)

    # get grid to sample pixels from feature map 
    search_size = self.corr_neighbour + self.patch_size - 1
    r_range = torch.arange(-(search_size // 2), search_size // 2 + 1, device=device).float()
    dy, dx = torch.meshgrid(r_range, r_range, indexing="ij")
    offsets = torch.stack([dx, dy], dim=-1) # [r_range, r_range, 2]

    center_coords_lv1 = target_pts_fls[:, [1, 0]].view(pts_num, 1, 1, 2) 

    grid1 = center_coords_lv1 + offsets.unsqueeze(0)
    norm_factor1 = torch.tensor([(self.fls_w - 1) / 2.0, (self.fls_h - 1) / 2.0], device=device)
    grid1 = (grid1 / norm_factor1) - 1.0

    center_coords_lv2 = center_coords_lv1 / self.encoder_downsize

    grid2 = center_coords_lv2 + offsets.unsqueeze(0)
    norm_factor2 = torch.tensor([(self.fls_w // self.encoder_downsize - 1) / 2.0, (self.fls_h // self.encoder_downsize - 1) / 2.0], device=device)
    grid2 = (grid2 / norm_factor2) - 1.0
    
    # get features patches from fmaps
    target_patches_fmap1 = F.grid_sample(self.fmap1[buff_target_frame_idx], grid1, mode='bilinear', padding_mode='zeros', align_corners=True) # shape [pts_num, C, r_corr, r_corr]
    target_patches_fmap2 = F.grid_sample(self.fmap2[buff_target_frame_idx], grid2, mode='bilinear', padding_mode='zeros', align_corners=True)

    # shape: (n, c, p+r-1, p+r-1) -> (n, c*p*p, r*r) , l - number of kernel poses 
    target_patches_fmap1 = F.unfold(target_patches_fmap1, kernel_size=(self.patch_size, self.patch_size), stride=1) # shape [pts_num, C*patch_size*patch_size. ]
    target_patches_fmap2 = F.unfold(target_patches_fmap2, kernel_size=(self.patch_size, self.patch_size), stride=1)

    target_patches_fmap1 = target_patches_fmap1.view(pts_num, self.fmap_c, self.patch_size*self.patch_size, self.corr_neighbour*self.corr_neighbour) # shape: (pts_num, c, patch_size^2, corr_neighbour^2)
    target_patches_fmap2 = target_patches_fmap2.view(pts_num, self.fmap_c, self.patch_size*self.patch_size, self.corr_neighbour*self.corr_neighbour)

    
    # --- get patches ---
    act_patches_f = self.patches_f[buff_source_frame_idx, local_patch_idx, :, :]
    # act_patches_c = self.patches_c[buff_source_frame_idx, local_patch_idx, :, :]
    act_patches_c = self.patches_c[buff_source_frame_idx, local_patch_idx, :]
    
    # --- calc correlation and connect to single tensor ---
    corr_map1 = torch.einsum('ncpr, ncp -> nr', target_patches_fmap1, act_patches_f)
    corr_map2 = torch.einsum('ncpr, ncp -> nr', target_patches_fmap2, act_patches_f)

    if pts_num > 0:
      corr_map = torch.cat((corr_map1.reshape(pts_num, -1), corr_map2.reshape(pts_num, -1)), dim= - 1)
    else:
      corr_map = torch.tensor([], device = device, dtype = torch.float)

    return corr_map, act_patches_c, i[valid_mask], j[valid_mask]
    
  # === define interface to obtain data === 

  
  def get_state(self):
    lcl_idx = self.frame_n % self.buff_size
    act_pose = self.poses[lcl_idx, :].detach().clone().cpu()
    act_time = self.time[lcl_idx].detach().clone().cpu()
    frame_num = self.frame_n
    return act_pose, act_time, frame_num 

  @property 
  def patch_coords_r_theta(self):
    return self.patch_state[:, :, :2]
  
  @property 
  def patch_coords_phi(self):
    return self.patch_state[:, :, 2:3]

  def update_state(self, opt_pose, opt_elev_angle, h, patch_idx):

    self.poses[:, :] = opt_pose.squeeze(0)

    self.patch_state[:, :, 2] = opt_elev_angle.squeeze(0).squeeze(-1)

    self.hidden_state[patch_idx // self.patches_per_frame, patch_idx % self.patches_per_frame, :] = h

    # self.patch_state[patch_idx // self.patches_per_frame, patch_idx % self.patches_per_frame, :] = opt_pts3d

  def get_hidden_state(self, patch_idx):

    h = self.hidden_state[patch_idx // self.patches_per_frame, patch_idx % self.patches_per_frame, :]

    return h 

  def append(self, frame, time_stamp, device):
    '''
    Add new frame and patches to graph. 
    Approximate new pose and create edges. 
    
    '''
    # --- extract patches --- 
    coords, patches_f, patches_c, fmap= self.patchifier(frame, mode =  self.patchifier_method)

    # --- add frame to graph ---
    self.add_frame(fmap, time_stamp, device)

    # --- add patches to graph ---
    self.add_patches(patches_f, patches_c, coords, device)

    # --- approximation of new initial pose ---
    self.approx_movement(device)

    # --- create edges for new data ---- 
    self.create_edges(device)
    
    # --- increment global frame idx ---
    self.frame_n += 1
    return
    
  def update_step(self, device):
    corr, ctx, valid_i, valid_j= self.corr(device)

    patch_idx = valid_i
    source_frame_idx = patch_idx // self.patches_per_frame
    target_frame_idx = valid_j
    
    return corr, ctx, source_frame_idx, target_frame_idx, patch_idx
  





  # def visu_data(self, source_frame_idx, target_frame_idx, patch_idx, last_frames = 3):

  #   # get source coordinates of patches (patch_idx)
  #   source_coords = self.patch_state[patch_idx // self.patches_per_frame % self.buff_size, patch_idx  % self.patches_per_frame]
    
  #   # project coordinates to target frames 
  #   source_poses = self.poses[source_frame_idx % self.buff_size]
  #   target_poses = self.poses[target_frame_idx % self.buff_size]
  #   target_coords = project_points(source_coords, source_poses, target_poses)

  #   # get only edges that are in range last_frames
  #   mask = (source_frame_idx >= self.frame_n - last_frames) & \
  #              (target_frame_idx >= self.frame_n - last_frames)
    
  #   visu_valid = mask.nonzero(as_tuple=True)[0]

  #   if len(visu_valid) == 0:
  #           return []
      
  #   # validate
  #   source_frame_idx = source_frame_idx[visu_valid]
  #   target_frame_idx = target_frame_idx[visu_valid]
  #   patch_idx = patch_idx[visu_valid]
  #   source_coords = source_coords[visu_valid]
  #   target_coords = target_coords[visu_valid]

  #   # get list of frames and coords for each patch
  #   unique_patches = torch.unique(patch_idx, sorted=True)

  #   output = []
  #   for target_patch_id in unique_patches:
  #     idxs = (patch_idx == target_patch_id).nonzero(as_tuple=True)[0]
  #     frames = torch.cat([source_frame_idx[idxs[0:1]], target_frame_idx[idxs]], dim = 0)
  #     coords = torch.cat([source_coords[idxs[0:1]], target_coords[idxs]], dim = 0)
  #     output.append((target_patch_id, frames, coords))
  #   return output