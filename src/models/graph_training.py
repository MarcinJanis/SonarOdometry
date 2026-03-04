import torch
import torch.nn as nn
import torch.nn.functional as F

import math 

from .patchifier import Patchifier
from .utils import hamilton_product, q_conjugate, project_points

# ================================================================================================== #
#
# Batch training mode graph. 
# This version of graph is only for traning purpose. 
# gets while package of frames in one moment, builds graph and allows to gradient flow to encoder. 
#
# ================================================================================================== #

class Graph(nn.Module):
  def __init__(self, model_cfg, sonar_cfg, training_cfg):
    super().__init__()

    # --- import training configuration
    self.batch_size = training_cfg.BATCH_SIZE
    self.frames_in_series = training_cfg.FRAMES_IN_SERIES

    # --- import sonar configuration ---
    self.r_min = sonar_cfg.range.min # min range
    self.r_max = sonar_cfg.range.max # max range

    self.fls_h = sonar_cfg.resolution.bins # vertical resolution of input fls image
    self.fls_w = sonar_cfg.resolution.beams # horizontal resolution of input fls image

    self.fov_vertical = sonar_cfg.fov.vertical * math.pi / 180 # vertical fov is in  [deg]
    self.fov_horizontal = sonar_cfg.fov.horizontal  * math.pi / 180 # horizontal fov is in [deg]

    # --- import sys configuration ---
    self.patches_per_frame = model_cfg.PATCHES_PER_FRAME # amount of patches generated per each frames
    self.patch_size = model_cfg.PATCH_SIZE # size of each patch, patch shape: (c, p, p)

    self.time_window = model_cfg.TIME_WINDOW # time window in frames history in which patches are tracked
    
    self.fmap_c = model_cfg.FEATURES_OUTPUT_CH # channels num of encoder output 
    self.cmap_c = model_cfg.CONTEXT_OUTPUT_CH 

    self.corr_neighbour = model_cfg.CORR_NEIGHBOUR # size of nieghbour of projected patch that is used in correlation calculations
    self.fmap_h = sonar_cfg.resolution.bins // model_cfg.ENCODER_DOWNSIZE # feature map size h
    self.fmap_w = sonar_cfg.resolution.beams // model_cfg.ENCODER_DOWNSIZE # feature map size w
    self.encoder_downsize = model_cfg.ENCODER_DOWNSIZE # encoder downsize factor 
    
    self.motion_model = model_cfg.MOTION_APPRO_MODEL # method used in initial estimating of new pose
    self.patchifier_method = model_cfg.PATCHIFIER_METHOD # method used in key points detection
    self.grid_size = (model_cfg.PATCHES_GRID_SIZE.y, model_cfg.PATCHES_GRID_SIZE.x) # grid size used to detect key points from whole image equally

    self.phi_init_mode = model_cfg.ELEVATION_INIT_MODE
    self.phi_init_min =  model_cfg.ELEVATION_ANGLE_INIT_MIN
    self.phi_init_max =  model_cfg.ELEVATION_ANGLE_INIT_MAX

    # # === Graph initialization ===
    # Note: in training version of graph, graph itself is created once, for whole batch so it is no initialized 
    self.time_stamp = None
    self.poses = None

    self.fmap1 = None
    self.fmap2 = None

    self.coords_r_theta = None # coords of patch, only r and theta, contants
    # coords_phi = None # coords of patch, only phi, which is estimating; self.patch_state = torch.cat([self.coords_r_theta, coords_phi], dim = 1)

    self.i = None
    self.j = None

    self.hidden_state = torch.zeros((self.batch_size*self.frames_in_series*self.patches_per_frame, self.cmap_c), dtype = torch.float)

    # --- Patchifier ---
    self.patchifier = Patchifier(model_cfg,
                                 debug_mode = False)
    
      
  def add_frame(self, fmap, time_stamp, device):
    b, n, c, h, w = fmap.shape
  
    self.fmap1 = fmap # F.avg_pool2d(fmap, 1, 1)
    
    self.fmap2 = F.avg_pool2d(fmap.view(b*n, c, h, w), self.encoder_downsize, self.encoder_downsize)
    self.fmap2 = self.fmap2.view(b, n, c, h // self.encoder_downsize, w // self.encoder_downsize)

    self.time_stamp = time_stamp
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

    b, n, p, d = coords.shape

    self.patches_f = patches_f # shape [b, n, patches_per_frame, c, psize*psize]     
    self.patches_c = patches_c # shape [b, n, patches_per_frame, c]     

    coords = coords.view(b*n*p, d)
    coords = self._scale_fls2phisical(coords)
    self.coords_r_theta = coords.view(b, n, p, d)

   
    if self.phi_init_mode == 'rand':
        coords_phi = torch.rand((b, n, p, 1), device=device, dtype=torch.float) * (self.phi_init_max - self.phi_init_min) + self.phi_init_min
    else: 
        coords_phi = torch.zeros((b, n, p, 1), device=device, dtype=torch.float) # init elevation angle with zeros

    coords_phi.requires_grad_(True)

    return coords_phi

  def approx_movement(self, device): #TODO: 

    if self.motion_model:
      poses = torch.zeros((self.batch_size, self.frames_in_series, 7), device=device, dtype=torch.float)
      poses[:, :, -1] = 1.0 # Quaternion w
      poses.requires_grad_(True)
    else:
      poses = torch.zeros((self.batch_size, self.frames_in_series, 7), device=device, dtype=torch.float)
      poses[:, :, -1] = 1.0 # Quaternion w
      poses.requires_grad_(True)
    
    return poses


  def create_edges(self, device):    

    i, j = [], [] # i - idx of patch, j - idx of frame. Connections: i -> j 
 
    for t in range(self.frames_in_series):
        for k in range(1, self.time_window + 1):
            if t - k >= 0:
                
                # edges: new patches -> old frames
                i.append(torch.arange(t * self.patches_per_frame, (t + 1) * self.patches_per_frame, device=device)) 
                j.append(torch.full((self.patches_per_frame,), t - k, device=device))
                
                # edges: old patches -> new frame
                i.append(torch.arange((t - k) * self.patches_per_frame, (t - k + 1) * self.patches_per_frame, device=device))
                j.append(torch.full((self.patches_per_frame,), t, device=device))

    # Concatenate edges 
    i_base = torch.cat(i, dim=0) if i else torch.tensor([], dtype=torch.long, device=device) # shape [2 * frame_in_series * patches_per_frame * time_window]
    j_base = torch.cat(j, dim=0) if j else torch.tensor([], dtype=torch.long, device=device) # shape [2 * frame_in_series * patches_per_frame * time_window]

    batch_indices = torch.arange(self.batch_size, device = device)# shape [batch_size, 1], [0, 1, 2, 3, 4]

    i_offsets = batch_indices * self.frames_in_series * self.patches_per_frame
    j_offsets = batch_indices * self.frames_in_series

    i_global = i_base.unsqueeze(0) + i_offsets.unsqueeze(-1)  # add with broadcasting 
    j_global = j_base.unsqueeze(0) + j_offsets.unsqueeze(-1)

    self.i = i_global.view(-1) # flat into one dimensional vector
    self.j = j_global.view(-1)

    return


  def corr(self, poses, coords_phi, eps, device):

    b, n, p, _ = self.coords_r_theta.shape

    # --- get source poses and target poses --- 
    source_frames_idx = self.i // self.patches_per_frame
    target_frames_idx = self.j

    poses = poses.view(b*n, 7) # shape: (batch_size, frames_in_series, 7) -> (batch_size * frames_in_series, 7)

    source_poses = poses[source_frames_idx]
    target_poses = poses[target_frames_idx]

    
    # --- reprojection ---
    coords_r_theta = self.coords_r_theta.view(b*n*p, 2)
    coords_phi = coords_phi.view(b*n*p, 1)

    patch_state = torch.cat([coords_r_theta, coords_phi], dim = 1) 
    source_coords  = patch_state[self.i]

    target_pts = project_points(source_coords, source_poses, target_poses)

    # --- edge validation --- 
    # find non valid edges
    theta_max = self.fov_horizontal / 2
    phi_max = self.fov_vertical / 2

    out_of_range = (target_pts[:,0] < (self.r_min - eps)) | (target_pts[:,0] > (self.r_max + eps))
    out_of_range = out_of_range | (torch.abs(target_pts[:,1]) > theta_max + eps)
    out_of_range = out_of_range | (torch.abs(target_pts[:,2]) > phi_max + eps)

    # discard non valid edges
    valid_mask = ~out_of_range
    target_pts = target_pts[valid_mask]
    valid_j = self.j[valid_mask]
    valid_i = self.i[valid_mask]

    pts_num = target_pts.shape[0]
    
    # --- get correlation neighbour from fmap --- 
    target_pts_fls = self._scale_phisical2fls(target_pts)

    # get grid to sample pixels from feature map 
    search_size = self.corr_neighbour + self.patch_size - 1
    r_range = torch.arange(-(search_size // 2), search_size // 2 + 1, device=device).float()
    dy, dx = torch.meshgrid(r_range, r_range, indexing="ij")
    offsets = torch.stack([dx, dy], dim=-1)

    center_coords_lv1 = target_pts_fls[:, [1, 0]].view(pts_num, 1, 1, 2) 

    grid1 = center_coords_lv1 + offsets.unsqueeze(0)
    norm_factor1 = torch.tensor([(self.fls_w - 1) / 2.0, (self.fls_h - 1) / 2.0], device=device)
    grid1 = (grid1 / norm_factor1) - 1.0

    center_coords_lv2 = center_coords_lv1 / self.encoder_downsize

    grid2 = center_coords_lv2 + offsets.unsqueeze(0)
    norm_factor2 = torch.tensor([(self.fls_w // self.encoder_downsize - 1) / 2.0, (self.fls_h // self.encoder_downsize - 1) / 2.0], device=device)
    grid2 = (grid2 / norm_factor2) - 1.0

    # get features patches from fmaps
    b, n, c, h, w = self.fmap1.shape
    fmap1_cpy, fmap2_cpy = self.fmap1.view(b*n, c, h, w), self.fmap2.view(b*n, c, h//self.encoder_downsize, w//self.encoder_downsize)

    target_patches_fmap1 = F.grid_sample(fmap1_cpy[valid_j], grid1, mode='bilinear', padding_mode='zeros', align_corners=True) #TODO:
    target_patches_fmap2 = F.grid_sample(fmap2_cpy[valid_j], grid2, mode='bilinear', padding_mode='zeros', align_corners=True)

    target_patches_fmap1 = F.unfold(target_patches_fmap1, kernel_size=(self.patch_size, self.patch_size), stride=1)
    target_patches_fmap2 = F.unfold(target_patches_fmap2, kernel_size=(self.patch_size, self.patch_size), stride=1)

    target_patches_fmap1 = target_patches_fmap1.view(pts_num, self.fmap_c, self.patch_size*self.patch_size, self.corr_neighbour*self.corr_neighbour)
    target_patches_fmap2 = target_patches_fmap2.view(pts_num, self.fmap_c, self.patch_size*self.patch_size, self.corr_neighbour*self.corr_neighbour)

    # --- get patches ---
    b, n, p, c1, d = self.patches_f.shape
    c2 = self.patches_c.shape[3]

    patches_f = self.patches_f.view(b*n*p, c1, d)
    patches_c = self.patches_c.view(b*n*p, c2)

    act_patches_f = patches_f[valid_i, :, :]
    act_patches_c = patches_c[valid_i, :]
    
    # --- calc correlation and connect to single tensor --- 
    corr_map1 = torch.einsum('ncpr, ncp -> nr', target_patches_fmap1, act_patches_f)
    corr_map2 = torch.einsum('ncpr, ncp -> nr', target_patches_fmap2, act_patches_f)

    corr_map = torch.cat((corr_map1.reshape(pts_num, -1), corr_map2.reshape(pts_num, -1)), dim=-1) 

    
    return corr_map, act_patches_c, valid_mask

  def get_hidden_state(self, patch_idx):
    
    return self.hidden_state[patch_idx, :]



  def append(self, frames, time_stamp, device):
    '''
    Add new frame and patches to graph. 
    Approximate new pose and create edges. 
    
    '''
    # self.N = frames.shape[0]

    # --- extract patches --- 
    coords, patches_f, patches_c, fmap = self.patchifier(frames, mode =  self.patchifier_method)

    # --- add frame to graph ---
    self.add_frame(fmap, time_stamp, device)

    # --- add patches to graph ---
    coords_phi = self.add_patches(patches_f, patches_c, coords, device)

    # --- approximation of new initial pose ---
    poses = self.approx_movement(device)

    # --- create edges for new data ---- 
    self.create_edges(device)

    return poses, coords_phi


  def update_step(self, poses, coords_phi,  device):

    # b, n, p, _ = coords_phi.shape

    corr, ctx, valid_mask = self.corr(poses, coords_phi, eps = 1e-2, device = device)
    
    patch_idx = self.i[valid_mask]
    source_frame_idx = patch_idx // self.patches_per_frame
    target_frame_idx = self.j[valid_mask]

    return corr, ctx, source_frame_idx, target_frame_idx, patch_idx


