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


class Graph(nn.Module):
  def __init__(self, window_size, fmap_dim, fmap_h, fmap_w, patch_size):
    super().__init__()
    # graph nodes:
    
    # frames graph buffors
    self.register_buffer('fg_poses', torch.zeros(window_size, 7)) # keeps estimated vehicle position in 6 DoF (x, y, z, q1, q2, q3, q4)
    self.register_buffer('fg_imap', torch.zeros(window_size, fmap_dim, fmap_h, fmap_w)) # keeps contex maps of frames in time window

    # patches graph buffors
    self.register_buffer('pg_xy', torch.zeros(0, 2))
    self.register_buffer('pg_frameidx', torch.zeros(0))
    self.register_buffer('pg_features', torch.zeros(0, fmap_dim*patch_size*patch_size))

    # graph edges:

    self.register_buffer('edge_i', torch.zeors(0, dtype=torch.int32)) # keeps idx of patch
    self.register_buffer('edge_j', torch.zeors(0, dtype=torch.int32)) # keeps idx of frame where patch is track
    self.register_buffer('edge_xy', torch.zeors(0, dtype=torch.int32)) # keeps local, 2d coordinates of patch i in frame j 


    def add_frame(self):
      pass

    def add_patch(self):
      pass

    

    
