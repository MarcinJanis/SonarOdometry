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
  def __init__(self, model_cfg, sonar_cfg):
    super().__init__()

    self.N = 7 # NUmber of frames in whole batch 

    # --- import sonar configuration ---
    self.r_min = sonar_cfg.range.min # min range
    self.r_max = sonar_cfg.range.max # max range

    self.fls_h = sonar_cfg.resolution.bins # vertical resolution of input fls image
    self.fls_w = sonar_cfg.resolution.beams # horizontal resolution of input fls image

    self.fov_vertical = sonar_cfg.fov.vertical * math.pi / 180 # vertical fov is in  [deg]
    self.fov_horizontal = sonar_cfg.fov.horizontal  * math.pi / 180# horizontal fov is in [deg]

    # --- import sys configuration ---
    self.patches_per_frame = model_cfg.PATCHES_PER_FRAME # amount of patches generated per each frames
    self.patch_size = model_cfg.PATCH_SIZE # size of each patch, patch shape: (c, p, p)

    self.time_window = model_cfg.TIME_WINDOW # time window in frames history in which patches are tracked
    
    self.fmap_c = model_cfg.FEATURES_OUTPUT_CH # channels num of encoder output 
    self.corr_neighbour = model_cfg.CORR_NEIGHBOUR # size of nieghbour of projected patch that is used in correlation calculations
    self.fmap_h = sonar_cfg.resolution.bins // model_cfg.ENCODER_DOWNSIZE # feature map size h
    self.fmap_w = sonar_cfg.resolution.beams // model_cfg.ENCODER_DOWNSIZE # feature map size w
    self.encoder_downsize = model_cfg.ENCODER_DOWNSIZE # encoder downsize factor 
    
    self.motion_model = model_cfg.MOTION_APPRO_MODEL # method used in initial estimating of new pose
    self.patchifier_method = model_cfg.PATCHIFIER_METHOD # method used in key points detection
    self.grid_size = (model_cfg.PATCHES_GRID_SIZE.y, model_cfg.PATCHES_GRID_SIZE.x) # grid size used to detect key points from whole image equally

    # # === Graph initialization ===
    # Note: in training version of graph, graph itself is created once, for whole batch so it is no initialized 
    self.time_stamp = None
    self.poses = None

    self.fmap1 = None
    self.fmap2 = None
    self.patch_state = None

    self.i = None
    self.j = None

    # --- Patchifier ---
    self.patchifier = Patchifier(model_cfg,
                                 debug_mode = False)
    
      
  def add_frame(self, fmap, time_stamp, device):

    self.fmap1 = F.avg_pool2d(fmap, 1, 1)
    self.fmap2 = F.avg_pool2d(fmap, self.encoder_downsize, self.encoder_downsize)

    self.time_stamp = time_stamp
    return 
  
      
  def _scale_fls2phisical(self, coords):
    '''
    Scale coords units from pixels in fls image to phisical units [m], [rad]
    
    :param coords: coords tensor, shape: (n, 2); n - points number, 2 - r and theta [pix]
    '''
    # range r - measured by sonar
    r_norm = coords[:, 1] / self.fls_h
    r = r_norm * (self.r_max - self.r_min) + self.r_min

    # azimuth angle theta - measured by sonar
    theta_norm = coords[:, 0] / self.fls_w - 0.5
    theta = theta_norm * self.fov_horizontal * torch.pi / 180.0
    
    return torch.stack([r, theta], dim = 1)

  def _scale_phisical2fls(self, coords):
    '''
    Scale coords units phisical units [m], [rad] to from pixels in fls image 
    
    :param coords: coords tensor, shape: (n, 2); n - points number, 2 - r [m] and theta [deg]
    '''
    # range r - measured by sonar
    r_norm = (coords[:, 0] - self.r_min) / (self.r_max - self.r_min)
    r = r_norm * self.fls_h

    # azimuth angle theta - measured by sonar
    theta_norm = coords[:, 1] * 180.0 / torch.pi / self.fov_horizontal 
    theta = (theta_norm + 0.5) * self.fls_w
    
    return torch.stack([r, theta], dim = 1)
  
  def add_patches(self, patches_f, patches_c, coords, device):
    self.patches_f = patches_f.view(-1, self.fmap_c, self.patch_size, self.patch_size)
    self.patches_c = patches_c.view(-1, self.fmap_c, self.patch_size, self.patch_size)

    patch_state_phi = torch.zeros((self.N * self.patches_per_frame, 1), device=device, dtype=torch.float) # init elevation angle with zeros
    patch_state_phi.requires_grad_(True)
    self.patch_state = torch.cat([self.coords, patch_state_phi], dim=-1) # shape: (total_frames, 3); coordinates: [r, theta, phi] 



  def approx_movement(self, device):
    '''
    In traning version of graph this function initialize poses for whole sequence
    mode: 
    - initialize with zeros
    - initialize first k frames with gt pose (?) 
    '''
    poses = torch.zeros((self.N, 7), device=device, dtype=torch.float)
    poses[:, -1] = 1.0 # Quaternion w
    poses.requires_grad_(True)
    
    self.poses = poses


  def create_edges(self, device):    
    '''
    Create set of edges in graph based on new frame and new patches.
    This set is for whole patch of N frames. 

    '''
    i, j = [], []
    
    for t in range(self.N):
        for k in range(1, self.time_window + 1):
            if t - k >= 0:
                # edges: new patches -> old frames
                i.append(torch.arange(t * self.patches_per_frame, (t + 1) * self.patches_per_frame, device=device)) 
                j.append(torch.full((self.patches_per_frame,), t - k, device=device))
                
                # edges: old patches -> new frame
                i.append(torch.arange((t - k) * self.patches_per_frame, (t - k + 1) * self.patches_per_frame, device=device))
                j.append(torch.full((self.patches_per_frame,), t, device=device))
                
    # Concatenate edges 
    self.i = torch.cat(i, dim=0) if i else torch.tensor([], dtype=torch.long, device=device)
    self.j = torch.cat(j, dim=0) if j else torch.tensor([], dtype=torch.long, device=device)
    return


  def corr(self, poses, device):

    device = poses.device

    # --- get source poses and target poses --- 
    source_frames_idx = self.i // self.patches_per_frame
    target_frames_idx = self.j

    source_poses = poses[source_frames_idx]
    target_poses = poses[target_frames_idx]

    # --- reprojection ---
    source_coords  = self.patch_state[self.i]
    target_pts = project_points(source_coords, source_poses, target_poses)

    # --- edge validation --- 
    # find non valid edges
    theta_max = self.fov_horizontal / 2
    phi_max = self.fov_vertical / 2

    out_of_range = (target_pts[:,0] < self.r_min) | (target_pts[:,0] > self.r_max)
    out_of_range = out_of_range | (torch.abs(target_pts[:,1]) > theta_max)
    out_of_range = out_of_range | (torch.abs(target_pts[:,2]) > phi_max)
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

    center_coords = target_pts_fls[:, [1, 0]].view(pts_num, 1, 1, 2) 
    grid1 = center_coords + offsets.unsqueeze(0)
    grid2 = center_coords + offsets.unsqueeze(0) * self.encoder_downsize

    norm_factor = torch.tensor([(self.fls_w - 1) / 2.0, (self.fls_h - 1) / 2.0], device=device)
    grid1 = (grid1 / norm_factor) - 1.0
    grid2 = (grid2 / norm_factor) - 1.0

    # get features patches from fmaps
    target_patches_fmap1 = F.grid_sample(self.fmap1[valid_j], grid1, mode='bilinear', padding_mode='zeros', align_corners=True)
    target_patches_fmap2 = F.grid_sample(self.fmap2[valid_j], grid2, mode='bilinear', padding_mode='zeros', align_corners=True)

    target_patches_fmap1 = F.unfold(target_patches_fmap1, kernel_size=(self.patch_size, self.patch_size), stride=1)
    target_patches_fmap2 = F.unfold(target_patches_fmap2, kernel_size=(self.patch_size, self.patch_size), stride=1)

    target_patches_fmap1 = target_patches_fmap1.view(pts_num, self.fmap_c, self.patch_size*self.patch_size, self.corr_neighbour*self.corr_neighbour)
    target_patches_fmap2 = target_patches_fmap2.view(pts_num, self.fmap_c, self.patch_size*self.patch_size, self.corr_neighbour*self.corr_neighbour)

    # --- get patches ---
    act_patches_f = self.patches_f[valid_i].view(pts_num, self.fmap_c, self.patch_size*self.patch_size)
    act_patches_c = self.patches_c[valid_i].view(pts_num, self.fmap_c, self.patch_size*self.patch_size)

    # --- calc correlation and connect to single tensor ---
    corr_map1 = torch.einsum('ncpr, ncp -> npr', target_patches_fmap1, act_patches_f)
    corr_map2 = torch.einsum('ncpr, ncp -> npr', target_patches_fmap2, act_patches_f)

    corr_map = torch.cat((corr_map1.reshape(pts_num, -1), corr_map2.reshape(pts_num, -1)), dim=-1) 

    return corr_map, act_patches_c, valid_mask


  def append(self, frames, time_stamp, device):
    '''
    Add new frame and patches to graph. 
    Approximate new pose and create edges. 
    
    '''
    # --- extract patches --- 
    coords, patches_f, patches_c, fmap = self.patchifier(frames, mode =  self.patchifier_method, device = device)

    # --- add frame to graph ---
    self.add_frame(fmap, time_stamp, device)

    # --- add patches to graph ---
    self.add_patches(patches_f, patches_c, coords, device)

    # --- approximation of new initial pose ---
    self.approx_movement(device)

    # --- create edges for new data ---- 
    self.create_edges(device)

    return


  def update_step(self, device):
    corr, ctx, val_idx = self.corr(device)
    return corr, ctx, val_idx