import torch
import torch.nn as nn
import torch.nn.functional as F

from .patchifier import Patchifier

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

      self.grid_size = (self.cfg.GRID_SIZE.y, self.cfg.GRID_SIZE.x)

      # --- Patchifier ---
      self.patchifier = Patchifier(patches_per_frame = self.patches_per_frame, 
                                   patch_size = self.patch_size, 
                                   grid_size = self.grid_size, 
                                   debug_mode = False)
    
      # --- poses buffers ---
      # self.register_buffer('time', torch.zeros(self.buff_size, dtype=torch.float)) # time stamp
      # self.register_buffer('poses', torch.zeros(self.buff_size, 7, dtype=torch.float)) # poses 

    
      # --- frame buffers ---

      self.frame_n = 0 # frame counter for ring buffer 
    
      self.register_buffer('fmap', torch.zeros((self.buff_size, self.fmap_h, self.fmap_w, self.fmap_c), dtype = torch.float)) # frames: features map 
      self.register_buffer('imap', torch.zeros((self.buff_size, self.fmap_h, self.fmap_w, self.fmap_c), dtype = torch.float)) # frames: context map 

      # --- patches buffers ---
      self.register_buffer('patches', torch.zeros((self.buff_size, self.patches_per_frame, self.patch_size, self.patch_size, self.fmap_c), dtype = torch.float)) # patches 

      # --- points 3D buffers ---
      self.register_buffer('points', torch.zeors((self.buff_size * self.patches_per_frame, 3), dtype = torch.float)) # points 3D (x, y, z) refered to patches
    
      window_size, fmap_dim, fmap_h, fmap_w, patch_size, patches_per_frame

    # self.register_buffer('edge_i', torch.zeros(0, dtype=torch.int32)) # keeps idx of patch
    # self.register_buffer('edge_j', torch.zeros(0, dtype=torch.int32)) # keeps idx of frame where patch is track
    # self.register_buffer('edge_xy', torch.zeros(0, dtype=torch.float)) # keeps local, 2d coordinates of patch i in frame j 


    def add_frame(self, imap, fmap): # zmienić: tutaj używac forward() seici do ekstrakcji oatchy -> zrobimy matriszkę, czyli nie przekawać jako argumenty, przekazać jedynie nową ramke
      
      # local index for ring buffer 
      local_idx = self.frame_n % self.buff_size
      # add context map to buffer 
      self.imap[local_idx, :, :, :] = imap.squeeze() # !!!!! sprawdzic czy squeeze ale prawdopodbnie trzeba pozbyc sie rozmiaru batcha
      # add features map to buffer 
      self.fmap[local_idx, :, :, :] = fmap.squeeze() # !!!!! sprawdzic czy squeeze ale prawdopodbnie trzeba pozbyc sie rozmiaru batcha
      # increment global frame index
      self.frame_n += 1
      
 
      # # approximate movement - constant speed model
      # self.fg_poses[-1,:] = self._approx_movement()

      # # assign idx to new frame
      # self.frame_idx += 1
      # self.fg_idx[-1] = self.frame_idx
      return 
      

    def add_patches(self):
      '''
      use patchifier, extract fmap, coords and inverse depth (probably elevation angle coord in my case)
      add to buffers
      
      '''
      pass

    def create_edges(self, nframes):
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
      # aprroximate new pose based on two previous poses 
      pass

    
    def forward(self, frame):
      '''
      execute all of above
      '''
      
    
