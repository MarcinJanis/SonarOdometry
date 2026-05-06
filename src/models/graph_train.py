import torch
import torch.nn as nn
import torch.nn.functional as F

import math 

from .patchifier import Patchifier
from .utils import project_points


class Graph(nn.Module):
    def __init__(self, model_cfg, sonar_cfg, batch_size, frames_in_series):
        super().__init__()
        
        # --- import sonar configuration ---
        self.r_min = sonar_cfg.range.min # min range
        self.r_max = sonar_cfg.range.max # max range

        self.fls_h = sonar_cfg.resolution.bins # vertical resolution of input fls image
        self.fls_w = sonar_cfg.resolution.beams # horizontal resolution of input fls image

        self.fov_vertical = sonar_cfg.fov.vertical # vertical fov [rad]
        self.fov_horizontal = sonar_cfg.fov.horizontal # horizontal fov [rad]
        
        # --- import sys configuration ---
        self.patches_per_frame = model_cfg.PATCHES_PER_FRAME # amount of patches generated per each frames
        self.patch_size = model_cfg.PATCH_SIZE # size of each patch, patch shape: (c, p, p)

        self.time_window = model_cfg.TIME_WINDOW # time window in frames history in which patches are tracked
        
        self.fmap_c = model_cfg.FEATURES_OUTPUT_CH # channels num of encoder output 
        self.cmap_c = model_cfg.CONTEXT_OUTPUT_CH # context features = hidden state features 

        self.corr_neighbour = model_cfg.CORR_NEIGHBOUR # size of nieghbour of projected patch that is used in correlation calculations
        self.fmap_h = sonar_cfg.resolution.bins // model_cfg.ENCODER_DOWNSIZE # feature map size h
        self.fmap_w = sonar_cfg.resolution.beams // model_cfg.ENCODER_DOWNSIZE # feature map size w
        
        self.encoder_downsize = model_cfg.ENCODER_DOWNSIZE # encoder downsize factor 
        self.fmap_downsize = model_cfg.FMAP_DOWNSIZE # encoder downsize factor 

        self.buff_size = model_cfg.BUFF_SIZE
        self.phi_init_mode = model_cfg.ELEVATION_INIT_MODE

        # --- import traning configuration --- 
        self.batch_size = batch_size
        self.frames_in_series = frames_in_series

        self.patchifier = Patchifier(model_cfg)
       
        # --- Dynamic graph init --- 
        self.i = torch.empty(0, dtype=torch.long) # Patches idx
        self.j = torch.empty(0, dtype=torch.long) # Target frame idx
        self.hidden_state = None # Hidden state
        
        self.frame_n = 0 # Frames cnte
        

    def extract_features(self, frames, device):

        coords, patches_f, patches_c, fmap = self.patchifier(frames) # extract features
        
        b, n, p, d = coords.shape

        coords_r_theta = self.scale_fls2phisical(coords.view(b*n*p, 2)) # coords of patches (r, theta)
        self.coords_r_theta = coords_r_theta.view(b, n, p, 2)
        
        coords_phi = self.init_phi(b, n, p, device) # coords of patches (phi) - init

        self.patches_f = patches_f # patches features
        self.patches_c = patches_c # patches context features

        self.fmap = fmap # frames feature map (orginal size)

        # b, n, c, h, w = fmap.shape
        # fmap2 = F.avg_pool2d(fmap.view(b*n, c, h, w), self.fmap_downsize, self.fmap_downsize)
        # self.fmap2 = fmap2.view(b, n, c, h // self.fmap_downsize, w // self.fmap_downsize) # frames feature map (downsized)

        return coords_phi, self.coords_r_theta

    def init_phi(self, b, n, p, device):
        if self.phi_init_mode == 'rand':
            coords_phi = (torch.rand((b, n, p, 1), device=device, dtype=torch.float) - 0.5) * self.fov_vertical # (self.phi_max - self.phi_min) + self.phi_min
        else: 
            coords_phi = torch.zeros((b, n, p, 1), device=device, dtype=torch.float) # init elevation angle with zeros
        return coords_phi


    def init_edges(self, init_frames, device):
        new_i, new_j = [], [] 
        for sf in range(init_frames): # for each source frame
            for tf in range(init_frames): # for each target frame
                # if sf - tf > 0: # in frames distance is smaller that time window
                if sf != tf: 
                    # edges: new patches -> old frames
                    new_i.append(torch.arange(sf * self.patches_per_frame, (sf + 1) * self.patches_per_frame, device=device)) 
                    new_j.append(torch.full((self.patches_per_frame,), tf, device=device))
                    
                    # edges: old patches -> new frame
                    new_i.append(torch.arange((tf) * self.patches_per_frame, (tf + 1) * self.patches_per_frame, device=device))
                    new_j.append(torch.full((self.patches_per_frame,), sf, device=device))

        i_base = torch.cat(new_i, dim=0)
        j_base = torch.cat(new_j, dim=0)

        # broadvast for batch size 
        batch_indices = batch_indices = torch.arange(self.batch_size, device=device)

        i_offsets = batch_indices * self.frames_in_series * self.patches_per_frame
        j_offsets = batch_indices * self.frames_in_series
        
        i_global = i_base.unsqueeze(0) + i_offsets.unsqueeze(-1) 
        j_global = j_base.unsqueeze(0) + j_offsets.unsqueeze(-1)

        self.i = i_global.view(-1)
        self.j = j_global.view(-1)

        # init hidden state
        new_edges = self.i.shape[0]
        self.hidden_state = torch.zeros((new_edges, self.cmap_c), device=device, dtype=torch.float)

        self.frame_n = init_frames - 1

    def create_new_edges(self, n, device):
        new_i, new_j = [], [] 
        
        start_tgt_frame = max(0, n - self.time_window)
        for tf in range(start_tgt_frame, n): # for each target frame in range time window      
                # edges: new patches -> old frames
                new_i.append(torch.arange(n * self.patches_per_frame, (n + 1) * self.patches_per_frame, device=device)) 
                new_j.append(torch.full((self.patches_per_frame,), tf, device=device))
                
                # edges: old patches -> new frame
                new_i.append(torch.arange(tf * self.patches_per_frame, (tf + 1) * self.patches_per_frame, device=device))
                new_j.append(torch.full((self.patches_per_frame,), n, device=device))

        if len(new_i) == 0:
            return # if there is no new edges
        
        i_base = torch.cat(new_i, dim=0)
        j_base = torch.cat(new_j, dim=0)

        # broadcast for batch
        batch_indices = batch_indices = torch.arange(self.batch_size, device=device)

        i_offsets = batch_indices * self.frames_in_series * self.patches_per_frame
        j_offsets = batch_indices * self.frames_in_series
        
        i_global = i_base.unsqueeze(0) + i_offsets.unsqueeze(-1) 
        j_global = j_base.unsqueeze(0) + j_offsets.unsqueeze(-1)

        new_i = i_global.view(-1)
        new_j = j_global.view(-1)

        # add to existing edges
        self.i = torch.cat([self.i, new_i], dim=0)
        self.j = torch.cat([self.j, new_j], dim=0)

        # add new hidden state
        new_edges = new_i.shape[0]
        new_hidden_state = torch.zeros((new_edges, self.cmap_c), device=device, dtype=torch.float)

        self.hidden_state = torch.cat([self.hidden_state, new_hidden_state], dim=0)

        # delete obsolete edges -> intraning mode, edges are not delted!
        valid_j_edges = (self.j >= (self.frame_n - self.buff_size)) # delete edges that points to obsolete frames
        
        src_frames_global = self.i // self.patches_per_frame
        valid_i_edges = (src_frames_global >= (self.frame_n - self.buff_size)) # delete edges that have source frame that dont exist any longer
        
        edges_to_keep = valid_j_edges & valid_i_edges

        self.i = self.i[edges_to_keep]
        self.j = self.j[edges_to_keep] 
        self.hidden_state = self.hidden_state[edges_to_keep]

        self.frame_n += 1


    
    def corr(self, poses, coords_phi, coords_eps, device):

        b, n, p, _ = self.coords_r_theta.shape
        n_act = poses.shape[1]

        # --- reproject points --- 

        # src and tgt framem idxs
        src_frame_idx = self.i // self.patches_per_frame
        tgt_frame_idx = self.j 

        # src poses, coords and tgt poses
        poses_flat = poses.view(b*n_act, 7)
 
        src_poses = poses_flat[src_frame_idx]
        tgt_poses = poses_flat[tgt_frame_idx]

        coords_r_theta = self.coords_r_theta.view(b*n*p, -1)
        coords_phi = coords_phi.view(b*n*p, -1)

        src_coords = torch.cat([coords_r_theta, coords_phi], dim=1)[self.i]

        # reproject
        tgt_cooords = project_points(src_coords, src_poses, tgt_poses)

        # --- edge validation ---

        theta_max = self.fov_horizontal / 2
        phi_max = self.fov_vertical / 2

        out_of_range = (tgt_cooords[:,0] < (self.r_min - coords_eps)) | (tgt_cooords[:,0] > (self.r_max + coords_eps))
        out_of_range = out_of_range | (torch.abs(tgt_cooords[:,1]) > theta_max + coords_eps)
        out_of_range = out_of_range | (torch.abs(tgt_cooords[:,2]) > phi_max + coords_eps)
        valid_mask = ~out_of_range

        valid_edges_num = self.i.shape[0] # all adges at this moment are treated as valid 

        # transform to fls values
        tgt_coords_val_fls = self.scale_phisical2fls(tgt_cooords)

        # --- get correlation of projected patches and target frame ---

        search_size = self.corr_neighbour + self.patch_size - 1 # search size around projected point

        # generate ofsets to search for each target coords center
        r_range = torch.arange(-(search_size // 2), search_size // 2 + 1, device=device).float()
        dy, dx = torch.meshgrid(r_range, r_range, indexing="ij")
        offsets = torch.stack([dx, dy], dim=-1)

        # add offsets to target coords
        center_traget_coords = tgt_coords_val_fls[:, [1, 0]] / self.encoder_downsize 
        center_target_coords = center_traget_coords.view(valid_edges_num, 1, 1, 2) # reshape for broadcasting

        # create sampling grid for fmap
        grid = center_target_coords + offsets.unsqueeze(0) # add offsets to coords center 
        norm_factor = torch.tensor([((self.fls_w / self.encoder_downsize) - 1) / 2.0, 
                                     ((self.fls_h / self.encoder_downsize)- 1) / 2.0], 
                                     device=device)
        grid = (grid / norm_factor) - 1.0
        # grid shape (E, S, S, 2), E - edges num, S - search size 

        b, n, c, h, w = self.fmap.shape
        fmap = self.fmap.view(b*n, c, h, w) 

        # --- sampling from fmap ---
        # To avoid creating copy of fmap for each patch (OOM), patches are grouped by common target frame,
        # and all selected patches are sampled from it once
        
        corr_neighbur_fmap_list = []
        orginal_indices = []

        unique_tgt_frame = torch.unique(self.j) # get all unique target frames

        for k in range(unique_tgt_frame.shape[0]): # for each tagret frame
            
            tgt_frame = unique_tgt_frame[k] # select target frame
            edge_mask = (self.j == tgt_frame) # select edges, that points to this target frame
            edges_act = torch.sum(edge_mask) # get number of selected edges

            orginal_indices.append(torch.nonzero(edge_mask).squeeze(-1)) # save indicies to restore
                     
            # get actual feature map and add batch size
            fmap_tgt = fmap[tgt_frame].unsqueeze(0) 
            
            # from sampling grid extract patches connected to this target frame
            grid_act = grid[edge_mask]

            # reshape grid and sample from fmap 
            # if edges_num size would be treated as batch size, fmap copy will be created to cut out this patch (instant OOM!)
            # To avoid it, edges_num size is connected to one of search_size dimension. 
            # Then cutting out patches is done from one fmap, as one huge patch with size (edges_act * search_size, search_size)
            # then patches are reshaped to proper size

            grid_reshaped = grid_act.view(1, edges_act * search_size, search_size, 2)
            sampled_patch = F.grid_sample(fmap_tgt, grid_reshaped, mode='bilinear', padding_mode='zeros', align_corners=True)
            corr_neighbur_fmap_list.append(sampled_patch.view(edges_act, c, search_size, search_size))

        # connect all sampled patches, and restor orginal order 
        corr_neighbur_fmap_unsorted = torch.cat(corr_neighbur_fmap_list, dim=0)

        cat_indices = torch.cat(orginal_indices, dim=0)
        reverse_order = torch.argsort(cat_indices)
        target_patches = corr_neighbur_fmap_unsorted[reverse_order] # shape (edges_num, c, search_size, search_size)
        
        # --- calculate correlation ---

        # batch matrix multiplication torch.bmm - muliplication whole batch of matrixes
        # (b, i, j) @ (b, m, n) = (b, i, n) 

        # reshape target patches to proper form 
        target_patches = target_patches.view(valid_edges_num, c, search_size*search_size)

        # expand patches for all active edges and reshape source patches to proper form 
        b, n, p, c1, d = self.patches_f.shape 
        source_patches = self.patches_f.view(b*n*p, c1, d)[self.i]
        source_patches = source_patches.view(valid_edges_num, c1, d).transpose(1, 2).contiguous()
        
        # source patches shape: (e, c, p*p) -> (e, p*p, c)
        # target patches shape: (e, c, s*s)
        correrlation = torch.bmm(source_patches, target_patches) 
        e, pp, ss = correrlation.shape
        # correlation shape: (e, p*p, s*s)

        # coorelation pyramid 
        corr_lv1 = correrlation.view(e, -1) # (e, p*p*s*s)

        corr_lv2 = F.avg_pool2d(correrlation.view(e*pp, 1, search_size, search_size), 
                                kernel_size=3, stride=2, padding=1)
        corr_lv2 = corr_lv2.view(e, -1)
        # pool2d is performed on grid (search_size x search_size) 
        # output shape: ((size + 2*pad - ksize) / stride) + 1
        # 
        # (13)


        # get context features for valid edges
        c2 = self.patches_c.shape[3]
        act_patches_c = self.patches_c.view(b*n*p, c2)[self.i, :]
        
        # connect correlation pyramid
        corr_map = torch.cat((corr_lv1, corr_lv2), dim=-1) 
        # corr map shape: (e, c), where:
        # c = p*p*s*s + p*p*x*x, x = (s - ksize + 2*paddig)/stride + 1

        return corr_map, act_patches_c, self.i, self.j, valid_mask.float()
    
    def get_hidden_state(self):
        return self.hidden_state #[valid_mask, :]

    def update_hidden_state(self, h):
        self.hidden_state = h

    def scale_fls2phisical(self, coords):

        # range r 
        r_norm = coords[:, 0] / self.fls_h
        r = r_norm * (self.r_max - self.r_min) + self.r_min

        # azimuth angle theta 
        theta_norm = coords[:, 1] / self.fls_w - 0.5
        theta = theta_norm * self.fov_horizontal 
        
        return torch.stack([r, theta], dim = 1)

    def scale_phisical2fls(self, coords):

        # range r 
        r_norm = (coords[:, 0] - self.r_min) / (self.r_max - self.r_min)
        r = r_norm * self.fls_h

        # azimuth angle theta 
        theta_norm = coords[:, 1] / self.fov_horizontal 
        theta = (theta_norm + 0.5) * self.fls_w
        
        return torch.stack([r, theta], dim = 1)
    
    def reset(self):
        '''Clear dynamic graph'''
        self.i = torch.empty(0, dtype=torch.long, device=self.i.device)
        self.j = torch.empty(0, dtype=torch.long, device=self.j.device)
        self.hidden_state = None
        self.frame_n = 0
