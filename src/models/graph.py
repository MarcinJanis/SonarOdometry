import torch
import torch.nn as nn
import torch.nn.functional as F

from .patchifier import Patchifier
from .utils import hamilton_product, q_conjugate

# from .lietorch import SE3

# Graf w DPVO =
# Element	Typ	Tensor
# Frame	węzeł	poses[frame_id]
# Patch	węzeł	patch_xy, idepth
# Patch → Frame	krawędź	edge_i, edge_j, coords


'''
Grap DPVO
Frame Node: 
  - frame (png)
  - frame id (int) -> Każda klatka ma unikalny? czy też id tyle ile klatek przechowujemy?
  
Patch Node: 
  - patch_xy - coords patcha  wklatce bazowej 
  - base_frame_id - id klatki bazowej 
  - inverse depth - najpierw initial (np. tak jak w poprzednie) 
  - features - z features extracor dla danego patcha 

Edges: 
  - i -> id of patch in patch graph
  - j -> id of frame 
  - coords -> przewidywana projekja punktu reprezentującego patch i na klatce j 
'''

# Ring buffers:
      # we operates on: buff_size - max size of buffer, and n - counter, buff
      # 1. Add new_record to ring buffer
      #   buff[n % buff_size] = new_record; n += 1
      # 2. Read from buffer from global index x
      #   record = buff[x % buff_size]
      # 3. Delete from buffer, something with global idx y, where y belongs to range [0, n - 1]
      #   indices = torch.arange(y, self.n - 1, device="cuda") # returns list of indexes
      #   target_idx = idices % buff_size 
      #   source_idx = (indices + 1) % buff_size
      #   buff[target_idx] = buff[source_idx] # -> to place with idx target_idx shift item with ind source_idx 
      #   n -= 1 # lower index, (on place n-1 there is a garbage) 


class Graph(nn.Module):
  def __init__(self, cfg):
    # window_size = actial window size + 1, to store new frame in buffor, before old frame will be delete
    super().__init__()
    self.cfg = cfg

    # --- import configuration ---
    self.buff_size = self.cfg.BUFF_SIZE
  
    self.patches_per_frame = self.cfg.PATCHES_PER_FRAME
    self.patch_size = self.cfg.PATCH_SIZE 
    
    self.fmap_c = self.cfg.FEATURES_OUTPUT_CH
  
    self.fmap_h = self.cfg.FLS_INPUT_HEIGHT // self.cfg.ENCODER_DOWNSIZE
    self.fmap_w = self.cfg.FLS_INPUT_WIDTH // self.cfg.ENCODER_DOWNSIZE

    self.motion_model = self.cfg.MOTION_APPRO_MODEL
   
    
    self.grid_size = (self.cfg.PATCHES_GRID_SIZE.y, self.cfg.PATCHES_GRID_SIZE.x)


    # --- Patchifier ---
    self.patchifier = Patchifier(self.cfg,
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
    self.register_buffer('patches', torch.zeros((self.buff_size, self.patches_per_frame,  self.fmap_c, self.patch_size, self.patch_size), dtype = torch.float)) # patches 

    # --- points 3D buffers ---
    self.register_buffer('points', torch.zeros((self.buff_size * self.patches_per_frame, 3), dtype = torch.float)) # points 3D (x, y, z) refered to patches
  
    # window_size, fmap_dim, fmap_h, fmap_w, patch_size, patches_per_frame

    self.register_buffer('edge_i', torch.zeros(0, dtype=torch.int32)) # keeps idx of patch
    self.register_buffer('edge_j', torch.zeros(0, dtype=torch.int32)) # keeps idx of frame where patch is track
    # self.register_buffer('edge_xy', torch.zeros(0, dtype=torch.float)) # keeps local, 2d coordinates of patch i in frame j 


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
      

  def add_patches(self, patches):
    # local index for ring buffer
    local_idx_min = self.frame_n % self.buff_size * self.patches_per_frame # powinno być tożsame z : self.frame_n*self.patches_per_frame % self.buff_size*self.patches_per_frame
    local_idx_max =  (self.frame_n + 1) % self.buff_size * self.patches_per_frame
    # add patches to graph 
    self.patches[local_idx_min:local_idx_max,:,:,:] = patches 
    return 
      
  def create_edges(self):

    # new patches -> new frame
    # new patches -> old frame
    # old patches -> new frame
    '''
    create edges: connect new patches with nlast frames:
      1) Add edge new patch with new frame (easy)
      2) for rach older frame, take its 6 dof pose, reproject point 2d -> 3d -> tranformation with pose -> 2d 
    calculating correlation betweenn patch and frame, assign coordinates with biggest correlation
    '''
    pass

  def garbage_grabber(self):
    '''
    delete patches with coordinates out the bounds from patch graph
    '''
    pass

  def _approx_movement(self):

    # if initialization
    if self.frame_n < 2:
      self.poses[k_idx, :] = torch.tensor([0,0,0,0,0,0,0], deveice = self.poses.device, dtype = self.poses.dtype)

    # indexes
    k_idx = self.frame_n % self.buff_size
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
    return x0
  
  # def forward(self, fmap, imap, patches, coords, time_stamp):
  def forward(self, frame, time_stamp):

    # !!! Patchifier on next level, here only add to graph and eventually run GRU
    # # exctract data from new frame
    # coords, patches, fmap, imap = self.patchifier(frame)
    # # coords.shape = [b, n, self.patches_per_frame, 2]
    # # patches.shape = [b, n, self.patches_per_frame, c, self.patch_size, self.patch_size]
    # # fmap.shape = [b, n, c, h, w]
    # # imap.shape = [b, n, c, h, w]
    
    # --- extract patches --- 
    coords, new_patches, fmap, imap = self.patchifier(frame, mode = 'harris')

    # --- add frame to graph ---
    self.add_frame(fmap, imap, time_stamp)

    # --- add patches to graph ---
    self.add_patches(new_patches)

    # --- approximation of new initial pose ---
    _ = self._approx_movement()

    # increment global frame idx 
    self.frame_n += 1
    
    # create graph edges
    return

  
