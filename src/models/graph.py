import torch
import torch.nn as nn
import torch.nn.functional as F

from .patchifier import Patchifier
from utils import hamilton_product

from .lietorch import SE3

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
      
      self.fmap_c = self.cfg.F_MAP_C
    
      self.fmap_h = self.cfg.F_MAP_H # <- change!
      self.fmap_w = self.cfg.F_MAP_W # <- change!

      self.motion_model = self.cfg.MOTION_MODEL
      self.motion_damping = self.cfg.MOTION_DAMPING
      self.grid_size = (self.cfg.GRID_SIZE.y, self.cfg.GRID_SIZE.x)

      # --- Patchifier ---
      self.patchifier = Patchifier(patches_per_frame = self.patches_per_frame, 
                                   patch_size = self.patch_size, 
                                   grid_size = self.grid_size, 
                                   debug_mode = False)
    
      # --- poses and time stamp buffers ---
      self.register_buffer('time', torch.zeros((self.buff_size), dtype=torch.float)) # time stamp
      self.register_buffer('poses', torch.zeros((self.buff_size, 7), dtype=torch.float)) # poses 

    
      # --- frame buffers ---

      self.frame_n = 0 # frame counter for ring buffer 
    
      self.register_buffer('fmap', torch.zeros((self.buff_size, self.fmap_h, self.fmap_w, self.fmap_c), dtype = torch.float)) # frames: features map 
      self.register_buffer('imap', torch.zeros((self.buff_size, self.fmap_h, self.fmap_w, self.fmap_c), dtype = torch.float)) # frames: context map 

      # --- patches buffers ---
      self.register_buffer('patches', torch.zeros((self.buff_size, self.patches_per_frame, self.patch_size, self.patch_size, self.fmap_c), dtype = torch.float)) # patches 

      # --- points 3D buffers ---
      self.register_buffer('points', torch.zeors((self.buff_size * self.patches_per_frame, 3), dtype = torch.float)) # points 3D (x, y, z) refered to patches
    
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
      self.patches[local_idx_min:local_idx_max,:,:,:]
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
        
        if self.motion_model == 'DAMPED_LINEAR':
          P1 = SE3(x1)
          P2 = SE3(x2)

          # To deal with varying camera hz
          fac = (t0-t1) / (t1-t2)
          xi = self.motion_damping * fac * (P1 * P2.inv()).log()
          new_pose = (SE3.exp(xi) * P1).data
        else:
          new_pose = x1
          
        self.poses[k_idx, :] = new_pose
        return new_pose
    
    def forward(self, fmap, imap, patches, coords, time_stamp):
    
      # !!! Patchifier on next level, here only add to graph and eventually run GRU
      # # exctract data from new frame
      # coords, patches, fmap, imap = self.patchifier(frame)
      # # coords.shape = [b, n, self.patches_per_frame, 2]
      # # patches.shape = [b, n, self.patches_per_frame, c, self.patch_size, self.patch_size]
      # # fmap.shape = [b, n, c, h, w]
      # # imap.shape = [b, n, c, h, w]
      
      # add frame to graph
      self.add_frame(fmap, imap, time_stamp)

      # add patches to graph
      self.add_patches(patches)

      # approximation of new initial pose 
      _ = self._approx_movement()

      # increment global frame idx 
      self.frame_n += 1
      
      # create graph edges
      

      
      '''
      execute all of above
      '''
      
    
