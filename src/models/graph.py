import torch
import torch.nn as nn
import torch.nn.functional as F

from .patchifier import Patchifier
from .utils import hamilton_product, q_conjugate, project_points


class Graph(nn.Module):
  def __init__(self, model_cfg, sonar_cfg):
    super().__init__()
    self.model_cfg = model_cfg
    self.sonar_cfg = sonar_cfg 

    # --- import sonar configuration ---
    self.r_min = self.sonar_cfg.range.min
    self.r_max = self.sonar_cfg.range.max

    self.fls_h = self.sonar_cfg.resolution.bins
    self.fls_w = self.sonar_cfg.resolution.beams

    self.fov_vertical = self.sonar_cfg.fov.vertical
    self.fov_horizontal = self.sonar_cfg.fov.horizontal

    # --- import sys configuration ---
    self.buff_size = self.model_cfg.BUFF_SIZE
  
    self.patches_per_frame = self.model_cfg.PATCHES_PER_FRAME
    self.patch_size = self.model_cfg.PATCH_SIZE 

    self.time_window = self.model_cfg.TIME_WINDOW
    
    self.fmap_c = self.model_cfg.FEATURES_OUTPUT_CH
    self.corr_neighbour = self.model_cfg.CORR_NEIGHBOUR
    self.fmap_h = self.sonar_cfg.resolution.bins // self.model_cfg.ENCODER_DOWNSIZE
    self.fmap_w = self.sonar_cfg.resolution.beams // self.model_cfg.ENCODER_DOWNSIZE
    self.encoder_downsize = self.model_cfg.ENCODER_DOWNSIZE
    
    self.motion_model = self.model_cfg.MOTION_APPRO_MODEL
   
    self.grid_size = (self.model_cfg.PATCHES_GRID_SIZE.y, self.model_cfg.PATCHES_GRID_SIZE.x)

    # --- Patchifier ---
    self.patchifier = Patchifier(self.model_cfg,
                                 debug_mode = False)
    
    # === Graph initialization ===
    self.frame_n = 0 # frame counter for ring buffer 

    # --- poses and time stamp buffers ---
    self.register_buffer('time', torch.zeros((self.buff_size), dtype=torch.float)) # time stamp
    self.register_buffer('poses', torch.zeros((self.buff_size, 7), dtype=torch.float)) # poses 

    # --- frame buffers ---
    self.register_buffer('fmap1', torch.zeros((self.buff_size, self.fmap_c, self.fmap_h, self.fmap_w), dtype = torch.float)) # frames: features map
    self.register_buffer('fmap2', torch.zeros((self.buff_size, self.fmap_c, self.fmap_h // self.encoder_downsize, self.fmap_w // self.encoder_downsize), dtype = torch.float)) # frames: features map 
    
    self.register_buffer('imap', torch.zeros((self.buff_size, self.fmap_c, self.fmap_h, self.fmap_w), dtype = torch.float)) # frames: context map 

    # --- patches buffers ---
    self.register_buffer('patches', torch.zeros((self.buff_size, self.patches_per_frame,  self.fmap_c, self.patch_size, self.patch_size), dtype = torch.float)) # patches features

    # --- patch center coords buffer ---
    self.register_buffer('patch_state', torch.zeros((self.buff_size, self.patches_per_frame, 3), dtype = torch.float)) # points (r, theta, phi) refered to patches + weight

    # --- source frame buffer ---
    self.register_buffer('source_frame', torch.zeros((self.buff_size, self.patches_per_frame), dtype = torch.int)) # id of source frame for each patch
                         
    # --- graphs edges --- 
    self.max_edges = self.buff_size * self.patches_per_frame * self.time_window * 2 # each patch (buff_size * patches_per_frame * 2) is connected to each frame in time window
    self.register_buffer('i', torch.zeros(self.max_edges, dtype=torch.int32)) # keeps idxs of patch
    self.register_buffer('j', torch.zeros(self.max_edges, dtype=torch.int32)) # keeps idxs of frame
    self.register_buffer('weights', torch.zeros(self.max_edges, dtype=torch.float)) # weights of each patch, how good estimation is, based on this patch 
    self.register_buffer('valid', torch.ones(self.max_edges, dtype=torch.bool)) # valid if its in range of frame, non valid if out of range 
  
  def add_frame(self, fmap, imap, time_stamp): 
    # local index for ring buffer 
    local_idx = self.frame_n % self.buff_size
    # add features map to buffer 
    self.fmap1[local_idx, :, :, :] = F.avg_pool2d(fmap.squeeze(0), 1, 1)
    self.fmap2[local_idx, :, :, :] = F.avg_pool2d(fmap.squeeze(0), self.encoder_downsize, self.encoder_downsize)
    
    # add context map to buffer 
    self.imap[local_idx, :, :, :] = imap.squeeze(0) # !!!!! sprawdzic czy squeeze ale prawdopodbnie trzeba pozbyc sie rozmiaru batcha
    # add time stamp 
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
    r_norm = coords[:, 0] - self.r_min / (self.r_max - self.r_min)
    r = r_norm * self.fls_h

    # azimuth angle theta - measured by sonar
    theta_norm = coords[:, 1] * 180.0 / torch.pi / self.fov_horizontal 
    theta = (theta_norm + 0.5) * self.fls_w
    
    return torch.stack([r, theta], dim = 1)
  
  def add_patches(self, patches, coords):
    # local index for ring buffer
    local_idx = self.frame_n % self.buff_size

    # add patches features to graph 
    self.patches[local_idx, :, :, :, :] = patches.squeeze(0) 

    # rescale coords to real world values  
    phisical_coords = self._scale_fls2phisical(coords.squeeze(0))
    r, theta = phisical_coords[:, 0], phisical_coords[:, 1]

    # approximate elevation angle - phi - to be optimized 
    phi = torch.zeros((self.patches_per_frame), device = coords.device, dtype = coords.dtype)
    
    
    # add to graph
    self.patch_state[local_idx, :, :] = torch.stack([r, theta, phi], dim=1)

    # add source frame id to graph
    self.source_frame[local_idx, :] = self.frame_n 

    return 

  def approx_movement(self):

    k_idx = self.frame_n % self.buff_size

    
    if self.frame_n < 2: # if initialization
      x0 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                                          device=self.poses.device, 
                                          dtype=self.poses.dtype)
    else: 
      # --- indexes ---
      k1_idx = (k_idx - 1) % self.buff_size  
      k2_idx = (k_idx - 2) % self.buff_size

      # --- get time stams ---
      t0 = self.time[k_idx]
      t1 = self.time[k1_idx]
      t2 = self.time[k2_idx]
      
      assert t0 != t1, f'[Error] Time stamps for frame {self.frame_n} and {self.frame_n - 1} are the same.\nMovement approximation is not possible.'
      assert t1 != t2, f'[Error] Time stamps for frame {self.frame_n - 1} and {self.frame_n - 2} are the same.\nMovement approximation is not possible.'
      
      # --- get previous position ---
      x1 = self.poses[k1_idx, :]
      x2 = self.poses[k2_idx, :]

      if self.motion_model == 'LINEAR':

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
        if s < 0.001: rot_axis = torch.tensor([1, 0, 0], device = diff.device, dtype = diff.dtype)
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
    self.poses[k_idx, :] = x0
    return 


  def create_edges(self):    
    
    # --- current patches -> past frame --- 
    new_patches = torch.arange(self.frame_n*self.patches_per_frame, (self.frame_n+1)*self.patches_per_frame) 
    past_frames = torch.arange(self.frame_n - 1, self.frame_n - 1- self.time_window, step=-1)

    i_new_patches = new_patches.repeat(self.time_window)
    j_past_frames = torch.repeat_interleave(past_frames, repeats=self.patches_per_frame)
    
    # --- past patches -> current frame --- 
    i_past_patches = torch.arange((self.frame_n - self.time_window)*self.patches_per_frame, self.frame_n*self.patches_per_frame) # <- here i finished
    j_new_frames = torch.ones(self.time_window*self.patches_per_frame) * self.frame_n 

    # --- concat --- 
    new_i = torch.cat((i_new_patches, i_past_patches), dim = 0)
    new_j = torch.cat((j_past_frames, j_new_frames), dim = 0)

    idx_low = (self.frame_n % self.buff_size) * self.patches_per_frame * self.time_window * 2
    idx_high = ((self.frame_n + 1) % self.buff_size) * self.patches_per_frame * self.time_window * 2

    self.i[idx_low:idx_high] = new_i
    self.j[idx_low:idx_high] = new_j

    self.weights[idx_low:idx_high] = torch.zeros((idx_high - idx_low))
    self.valid[idx_low:idx_high] = torch.ones((idx_high - idx_low))

    return 
  
  def corr(self):

    # ---
    device = self.fmap1.device 
    
    # --- choose only valid edges ---
    # valid edge: projected points are in range of fmap and depts in not negative 
    i_val = self.i[self.valid].long()
    j_val = self.j[self.valid].long()

    # --- get source poses and target poses
    source_frames_idx = i_val // self.patches_per_frame
    local_patch_idx = i_val % self.patches_per_frame

    buff_source_frame_idx = source_frames_idx % self.buff_size
    buff_target_frame_idx = j_val % self.buff_size

    source_poses = self.poses[buff_source_frame_idx, :]
    target_poses = self.poses[buff_target_frame_idx, :]

    source_coords = self.patch_state[buff_source_frame_idx, local_patch_idx, :]
    
    # --- reprojection ---
    target_pts = project_points(source_coords, source_poses, target_poses)

    # --- edges validation --- 
    theta_max = self.fov_horizontal / 2
    phi_max = self.fov_vertical / 2

    out_of_range = (target_pts[:,0] < self.r_min) | (target_pts[:,0] > self.r_max)
    out_of_range = out_of_range | (torch.abs(target_pts[:,1]) > theta_max)
    out_of_range = out_of_range | (torch.abs(target_pts[:,2]) > phi_max)

    active_edges_idx = torch.nonzero(self.valid, as_tuple=True)[0]
    invalid_edges_idx = active_edges_idx[out_of_range]
    self.valid[invalid_edges_idx] = 0

    # delete non-valid patches - check!?
    target_pts = target_pts[~out_of_range]
    buff_source_frame_idx = buff_source_frame_idx[~out_of_range]
    buff_target_frame_idx = buff_target_frame_idx[~out_of_range]
    local_patch_idx = local_patch_idx[~out_of_range]

    pts_num, _ = target_pts.shape 
    
    # --- get correlation neighbour from fmap --- 
    target_pts = self._scale_phisical2fls(target_pts)
    # add offsets
    r = torch.arange(-self.corr_neighbour, self.corr_neighbour + 1, device=device) # for r_corr = 2, r = [-2, -1, 0, 1, 2]
    dy, dx = torch.meshgrid(r, r, indexing="ij") # dy = [-2, -1, 0, 1, 2], dx =  [-2, -1, 0, 1, 2]
    
    offsets = torch.stack([dx, dy], dim=-1).float() # shape [r_corr, r_corr, 2]

    offset_norm = torch.zeros_like(offsets).unsqueeze(0) # [1, r_corr, r_cor, 2]
    offset_norm[..., 0] = 2.0 * offsets[..., 0] / (self.fls_w - 1)
    offset_norm[..., 1] = 2.0 * offsets[..., 1] / (self.fls_h - 1)

    
    grid = target_pts.view(pts_num, 1, 1, 2) + offsets.unsqueeze(0)
    grid_norm = torch.zeros_like(grid)
    grid_norm[..., 0] = 2.0 * grid[..., 0] / (self.fls_w - 1) - 1.0 # norm x
    grid_norm[..., 1] = 2.0 * grid[..., 1] / (self.fls_h - 1) - 1.0 # norm y 

    grid_norm1 = grid_norm + offset_norm
    grid_norm2 = grid_norm + offset_norm * self.encoder_downsize
    
    # get features patches from maps
    target_patches_fmap1 = F.grid_sample(self.fmap1[buff_target_frame_idx], grid_norm1, mode='bilinear', padding_mode='zeros', align_corners=True) # shape [pts_num, C, r_corr, r_corr]
    target_patches_fmap2 = F.grid_sample(self.fmap2[buff_target_frame_idx], grid_norm2, mode='bilinear', padding_mode='zeros', align_corners=True)

    # target_patches = torch.stack((target_patches_fmap1.unsqueeze(-1), target_patches_fmap2.usnqueeze(-2)), dim = -1)

    # --- calculate correlation ---
    
    
        
    return target_pts, buff_source_frame_idx, buff_target_frame_idx, local_patch_idx
    

  
  # === define interface to obtain data === #TODO
  @property
  def state(self):
    pose_vct = self.pose.detatch().cpu()
    time_vct = self.time.detatch().cpu()
    frame_num = self.frame_n
    return pose_vct, time_vct, frame_num 
    
  @property
  def last_pose(self):
    return self.pose[(self.frame_n - 1)%self.buff_size].detatch().cpu()

  @property
  def id(self):
    return self.frame_n
    
  @property
  def shape(self):
    size_dict = {
      'time':self.time.shape,
      'poses':self.poses.shape,
      'fmap1':self.fmap1.shape,
      'fmap2':self.fmap2.shape,
      'imap':self.imap.shape,
      'patches':self.patches.shape,
      'patch_state':self.patch_state.shape,
      'source_frame':self.source_frame.shape,
      'i':self.i.shape,
      'j':self.j.shape,
      'weight':self.weight.shape,
      'valid':self.valid.shape,
    }
  
  def forward(self, frame, time_stamp):
    
    # --- extract patches --- 
    coords, new_patches, fmap, imap = self.patchifier(frame, mode = 'harris')

    # --- add frame to graph ---
    self.add_frame(fmap, imap, time_stamp)

    # --- add patches to graph ---
    self.add_patches(new_patches, coords)

    # --- approximation of new initial pose ---
    self.approx_movement()

    # --- create edges for new data ---- 
    self.create_edges()
    
    # --- increment global frame idx ---
    self.frame_n += 1

    # --- calculate correlation ---- 
    self.corr()

    return




#### === Testy:
'''
W osobnym pliku ipynb
1. Sprawdzić rozmiary grafu
2. Sprawdzić dodawanie łatek i frameów, np. wygenerowac mapę cech składającą się tylko z 1 na powierzchni gdzie ma być jakiś patch, 
   2 gdzie ma być kolejny itd i zonaczyć czy będą dobrze przypisane
3. Dodać sprawdzanie aproksymacji ruchu
3. Wypisać krawędzie dla jakiś nie dużych rozmiarów i zobaczyć czy indkesy łączone są odpowiednio "każdy z każdym"


'''




'''

global_frame_idx -> buff_frame_idx gfi % buff_size


global_patch_idx -> global_frame_idx: gpi // patches_per_frame
global_patch_idx -> local_frame_idx: gpi % patches_per_frame

'''
