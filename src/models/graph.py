import torch
import torch.nn as nn
import torch.nn.functional as F

from .patchifier import Patchifier
from .utils import hamilton_product, q_conjugate, project_points


class Graph(nn.Module):
  def __init__(self, model_cfg, sonar_cfg):
    # window_size = actial window size + 1, to store new frame in buffor, before old frame will be delete
    super().__init__()
    self.model_cfg = model_cfg
    self.sonar_cfg = sonar_cfg 

    # --- import sonar configuration ---
    self.r_min = self.sonar_cfg.range.min
    self.r_max = self.sonar_cfg.range.max

    self.fls_h = sonar_cfg.resolution.bins
    self.fls_w = self.sonar_cfg.resolution.beams

    self.fov_vertical = sonar_cfg.fov.vertical
    self.fov_horizontal = sonar_cfg.fov.horizontal

    # --- import sys configuration ---
    self.buff_size = self.model_cfg.BUFF_SIZE
  
    self.patches_per_frame = self.model_cfg.PATCHES_PER_FRAME
    self.patch_size = self.model_cfg.PATCH_SIZE 

    self.time_window = self.model_cdg.TIME_WINDOW
    
    self.fmap_c = self.model_cfg.FEATURES_OUTPUT_CH
  
    self.fmap_h = self.sonar_cfg.resolution.bins // self.model_cfg.ENCODER_DOWNSIZE
    self.fmap_w = self.sonar_cfg.resolution.beams // self.model_cfg.ENCODER_DOWNSIZE

    self.motion_model = self.model_cfg.MOTION_APPRO_MODEL
   
    self.grid_size = (self.model_cfg.PATCHES_GRID_SIZE.y, self.model_cfg.PATCHES_GRID_SIZE.x)

    # --- Patchifier ---
    self.patchifier = Patchifier(self.model_cfg,
                                 debug_mode = False)
    
    # --- Graph initialization ---
    self.frame_n = 0 # frame counter for ring buffer 

    # --- poses and time stamp buffers ---
    self.register_buffer('time', torch.zeros((self.buff_size), dtype=torch.float)) # time stamp
    self.register_buffer('poses', torch.zeros((self.buff_size, 7), dtype=torch.float)) # poses 

    # --- frame buffers ---
    self.register_buffer('fmap', torch.zeros((self.buff_size, self.fmap_c, self.fmap_h, self.fmap_w), dtype = torch.float)) # frames: features map 
    self.register_buffer('imap', torch.zeros((self.buff_size, self.fmap_c, self.fmap_h, self.fmap_w), dtype = torch.float)) # frames: context map 

    # --- patches buffers ---
    self.register_buffer('patches', torch.zeros((self.buff_size, self.patches_per_frame,  self.fmap_c, self.patch_size, self.patch_size), dtype = torch.float)) # patches features

    # --- patch center coords buffer ---
    self.register_buffer('patch_state', torch.zeros((self.buff_size, self.patches_per_frame, 3), dtype = torch.float)) # points (r, theta, phi) refered to patches + weight

    # --- source frame buffer ---
    self.register_buffer('source_frame', torch.zeros((self.buff_size, self.patches_per_frame), dtype = torch.int)) # id of source frame for each patch
                         

    # --- graphs edges --- 
    max_edges = self.buff_size * self.patches_per_fram * self.time_window # each patch (buff_size * patches_per_frame) is connected to each frame in time window
    self.register_buffer('i', torch.zeros(max_edges, dtype=torch.int32)) # keeps idxs of patch
    self.register_buffer('j', torch.zeros(max_edges, dtype=torch.int32)) # keeps idxs of frame
    self.register_buffer('weights', torch.zeros(max_edges, dtype=torch.float)) # weights of each patch, how good estimation is, based on this patch 
    self.register_buffer('valid', torch.zeros(max_edges, dtype=torch.bool)) # valid if its in range of frame, non valid if out of range 
  
  def add_frame(self, fmap, imap, time_stamp): 
    # local index for ring buffer 
    local_idx = self.frame_n % self.buff_size
    # add features map to buffer 
    self.fmap[local_idx, :, :, :] = fmap.squeeze(0) # !!!!! sprawdzic czy squeeze ale prawdopodbnie trzeba pozbyc sie rozmiaru batcha
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
    
    return r, theta 

  def _scale_phisical2fls(self, coords):

    # range r - measured by sonar
    r_norm = coords[:, 0] - self.r_min / (self.r_max - self.r_min)
    r = r_norm * self.fls_h

    # azimuth angle theta - measured by sonar
    theta_norm = coords[:, 1] * 180.0 / torch.pi / self.fov_horizontal 
    theta = (theta_norm + 0.5) * self.fls_w
    
    return r, theta 
  
  def add_patches(self, patches, coords):
    # local index for ring buffer
    local_idx = self.frame_n % self.buff_size

    # add patches features to graph 
    self.patches[local_idx, :, :, :, :] = patches.squeeze(0) 

    # rescale coords to real warold values  
    r, theta = self._scale_fls2phisical(coords.squeeze(0))

    # approximate elevation angle - phi - to be optimized 
    phi = torch.zeros((self.patches_per_frame), device = coords.device, dtype = coords.dtype)
    
    
    # add to graph
    self.patch_state[local_idx, :, :] = torch.stack([r, theta, phi], dim=1)

    # add source frame id to graph
    self.source_frame[local_idx, :] = self.frame_n 

    return 

  def _approx_movement(self):

    k_idx = self.frame_n % self.buff_size

    
    if self.frame_n < 2: # if initialization
      x0 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                                          device=self.poses.device, 
                                          dtype=self.poses.dtype)
    else: 
      # indexes
      k1_idx = (k_idx - 1) % self.buff_size  
      k2_idx = (k_idx - 2) % self.buff_size

      # get time stams
      t0 = self.time[k_idx]
      t1 = self.time[k1_idx]
      t2 = self.time[k2_idx]

      # get previous position
      x1 = self.poses[k1_idx, :]
      x2 = self.poses[k2_idx, :]

      if self.motion_model == 'LINEAR':

        # linear displacement
        new_pose = x1[0:3] + (x1[0:3] - x2[0:3])/(t1 - t2)*(t0 - t1) 

        # quaterions
        q1 = x1[3:]
        q2 = x2[3:]

        # find shortest rotation 
        dot = (q1 * q2).sum() 
        if dot < 0: q1 = -q1

        # rotation - quaterions difference
        q2_conj = q_conjugate(q2)
        diff = hamilton_product(q1, q2_conj)  # diff q2 -> q1: diff = q2 * q1^-1

        # rotation axis 
        s = torch.sqrt(torch.clamp(1 - diff[-1]*diff[-1], 0.0))
        if s < 0.001: rot_axis = torch.tensor([1, 0, 0], device = diff.device, dtype = diff.dtype)
        else: rot_axis = diff[:-1]/s

        # rotation angle
        rot_angle = 2 * torch.acos(torch.clamp(diff[-1], -1.0, 1.0))
        rot_angle_appro = rot_angle/(t1 - t2)*(t0 - t1) # approximation apriori, to t0

        # apply rotation
        q_step_vect = rot_axis * torch.sin(rot_angle_appro/2)
        q_step_scal = torch.cos(rot_angle_appro/2).unsqueeze(0)   
        q_step = torch.cat((q_step_vect, q_step_scal), dim=0)

        q0 = hamilton_product(q_step, q1)
        q0 = q0 /torch.norm(q0)

        x0 = torch.cat((new_pose, q0), dim=0)

      else:
        x0 = x1

    # save to buffer
    self.poses[k_idx, :] = x0
    return 


  def _create_edges(self):    
    # TODO: add device 
    # --- current patches -> past frame --- 
    new_patches = torch.arrange(self.frame_n*self.patches_per_frame, (self.frame_n+1)*self.patches_per_frame)
    past_frames = torch.arrange(self.frame_n - 1, self.frame_n - 1- self.time_window, step=-1)

    i_new_patches = new_patches.repeat(self.time_window)
    j_past_frames = torch.repeat_interleave(past_frames, repeats=self.patches_per_frame)
    
    # --- past patches -> current frame --- 
    i_past_patches = torch.arrange((self.frames_n - self.time_window)*self.patches_per_frame, self.frames_n*self.patches_per_frame) # <- here i finished
    j_current_frames = torch.ones(self.time_window*self.patches_per_frame) * self.frame_n 

    # --- concat --- 
    new_i = torch.cat((i_new_patches, i_past_patches), dim = 0)
    new_j = torch.cat((j_past_frames, j_current_frames), dim = 0)

    idx_low = (self.frames_n % self.buff_size) * self._patches_per_frame * self.time_window
    idx_high = ((self.frames_n + 1) % self.buff_size) * self._patches_per_frame * self.time_window
    
    self.i[idx_low:idx_high] = new_i
    self.j[idx_low:idx_high] = new_j
    self.weights[idx_low:idx_high] = 0.0
    self.valid[idx_low:idx_high] = 1.0

    return 
  
  def forward(self, frame, time_stamp):
    
    # --- extract patches --- 
    coords, new_patches, fmap, imap = self.patchifier(frame, mode = 'harris')

    # --- add frame to graph ---
    self.add_frame(fmap, imap, time_stamp)

    # --- add patches to graph ---
    self.add_patches(new_patches, coords)

    # --- approximation of new initial pose ---
    self._approx_movement()

    # --- create edges for new data ---- 
    self._create_edges()
    # --- increment global frame idx ---
    self.frame_n += 1

    return 



# TODO next:

# 1. Krawędzie
# - przypisać każdy patch do swojej ramki (???)                                 
# - przypisyać nowe patche do ramki z przeszłości (bierzmy tu jakięś okno czaosowe pod uwagę np. 4 czy 5 w tył)
# - dla patchy ze strych ramek przypsiać je do nowej ramki 

# 2. Pętla 

# w pętli: 
# - dla każej z krawędzi
# - biorę patch i jego coords, transformuję do 3D
# - przesuwam o zmianę pozycji kamery (pose z tamtej klatki i pose bierzący)
# - reprojekcja na wspólrzędnę fls
# - liczę korelację w danym miejscu oraz korelację w okół tego miejsca. 
# - w efekcie dla każdej krawędzi otrzymję siatkę wartości korelacji np. 3x3

