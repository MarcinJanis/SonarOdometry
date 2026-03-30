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
        
        self.phi_max =  - sonar_cfg.position.pitch # max available elevation angle
        self.phi_min =  - sonar_cfg.position.pitch - self.fov_vertical # min available elevation angle
        
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

        self.fmap1 = fmap # frames feature map (orginal size)

        b, n, c, h, w = fmap.shape
        fmap2 = F.avg_pool2d(fmap.view(b*n, c, h, w), self.encoder_downsize, self.encoder_downsize)
        self.fmap2 = fmap2.view(b, n, c, h // self.encoder_downsize, w // self.encoder_downsize) # frames feature map (downsized)

        return coords_phi, self.coords_r_theta

    def init_phi(self, b, n, p, device, mode = 'rand'):
        if self.phi_init_mode == 'rand':
            coords_phi = torch.rand((b, n, p, 1), device=device, dtype=torch.float) * (self.phi_max - self.phi_min) + self.phi_min
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
                    new_j.append(torch.full((self.patches_per_frame,), sf-tf, device=device))
                    
                    # edges: old patches -> new frame
                    new_i.append(torch.arange((sf - tf) * self.patches_per_frame, (sf - tf + 1) * self.patches_per_frame, device=device))
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

        out_of_range = (tgt_cooords[:,0] < (self.r_min - coords_eps)) | (tgt_cooords[:,0] > (self.r_max + coords_eps))
        out_of_range = out_of_range | (torch.abs(tgt_cooords[:,1]) > theta_max + coords_eps)
        out_of_range = out_of_range | (tgt_cooords[:,2] > self.phi_max + coords_eps)
        out_of_range = out_of_range | (tgt_cooords[:,2] < self.phi_min - coords_eps)
        valid_mask = ~out_of_range

        # discard non valid edges 
        tgt_cooords_val = tgt_cooords[valid_mask]
        i_val = self.i[valid_mask]
        j_val = self.j[valid_mask]

        valid_edges_num = i_val.shape[0]

        if valid_edges_num == 0: 
            return None, None, i_val, j_val, valid_mask

        # transform to fls values
        tgt_coords_val_fls = self.scale_phisical2fls(tgt_cooords_val)

        # --- get correlation of projected patches and target frame ---

                
        search_size = self.corr_neighbour + self.patch_size - 1

        # ofsets to search for each target coords
        r_range = torch.arange(-(search_size // 2), search_size // 2 + 1, device=device).float()
        dy, dx = torch.meshgrid(r_range, r_range, indexing="ij")
        offsets = torch.stack([dx, dy], dim=-1)

        # add offsets to target coords
        center_coords_lv1 = tgt_coords_val_fls[:, [1, 0]].view(valid_edges_num, 1, 1, 2) 

        # create normal size sampling grid 
        grid1 = center_coords_lv1 + offsets.unsqueeze(0)
        norm_factor1 = torch.tensor([(self.fls_w - 1) / 2.0, (self.fls_h - 1) / 2.0], device=device)
        grid1 = (grid1 / norm_factor1) - 1.0 # norm to 0 - 1, cenetr in 0.5

        # downsample target coords with offsets
        center_coords_lv2 = center_coords_lv1 / self.encoder_downsize

        # create downsample sampling grid
        grid2 = center_coords_lv2 + offsets.unsqueeze(0)
        norm_factor2 = torch.tensor([(self.fls_w // self.encoder_downsize - 1) / 2.0, (self.fls_h // self.encoder_downsize - 1) / 2.0], device=device)
        grid2 = (grid2 / norm_factor2) - 1.0

        # get features patches from fmaps 
        b, n, c, h, w = self.fmap1.shape

        fmap1_cpy = self.fmap1.view(b*n, c, h, w)
        fmap2_cpy = self.fmap2.view(b*n, c, h//self.encoder_downsize, w//self.encoder_downsize)

        target_patches_fmap1 = F.grid_sample(fmap1_cpy[j_val], grid1, mode='bilinear', padding_mode='zeros', align_corners=True)
        target_patches_fmap2 = F.grid_sample(fmap2_cpy[j_val], grid2, mode='bilinear', padding_mode='zeros', align_corners=True)
        # shapes: [valid_edges_num, fmap_c, search_size, search_size]

        # represent in standard conv form (batch, channels, h, w)
        target_patches_fmap1 = target_patches_fmap1.view(1, valid_edges_num * self.fmap_c, search_size, search_size)
        target_patches_fmap2 = target_patches_fmap2.view(1, valid_edges_num * self.fmap_c, search_size, search_size)

        # represent each patch as conv kernel
        b, n, p, c1, d = self.patches_f.shape
        c2 = self.patches_c.shape[3]

        patch_features_kernel = self.patches_f.view(b*n*p, c1, self.patch_size, self.patch_size)[i_val, :, :, :] # patches features kernel

        # perform conv2d with group = valid_edges_num
        # each of kernels, represnting each patch have acces to features of corresponding fmap features
        corr_map1 = F.conv2d(target_patches_fmap1, patch_features_kernel, groups=valid_edges_num)
        corr_map2 = F.conv2d(target_patches_fmap2, patch_features_kernel, groups=valid_edges_num)
        # output shape: (1, valid_edges_num, corr_neighbour, corr_neighbour)

        # get context features for valid edges
        act_patches_c = self.patches_c.view(b*n*p, c2)[i_val, :]
        
        # calc correlation and connect to single tensor 
       
        corr_map = torch.cat((corr_map1.view(valid_edges_num, -1), corr_map2.view(valid_edges_num, -1)), dim=-1) 

        return corr_map, act_patches_c, i_val, j_val, valid_mask
    

    # === obsolete version of corr fcn === 
    # def corr(self, poses, coords_phi, coords_eps, device):
        
    #     b, n, p, _ = self.coords_r_theta.shape
    #     n_act = poses.shape[1]
    #     # --- reproject points --- 

    #     # src and tgt framem idxs
    #     src_frame_idx = self.i // self.patches_per_frame
    #     tgt_frame_idx = self.j 

    #     # src poses, coords and tgt poses
        
    #     poses_flat = poses.view(b*n_act, 7)
 
    #     src_poses = poses_flat[src_frame_idx]
    #     tgt_poses = poses_flat[tgt_frame_idx]

    #     coords_r_theta = self.coords_r_theta.view(b*n*p, -1)
    #     coords_phi = coords_phi.view(b*n*p, -1)

    #     src_coords = torch.cat([coords_r_theta, coords_phi], dim=1)[self.i]

    #     # reproject
    #     tgt_cooords = project_points(src_coords, src_poses, tgt_poses)

    #     # --- edge validation ---
    #     theta_max = self.fov_horizontal / 2

    #     out_of_range = (tgt_cooords[:,0] < (self.r_min - coords_eps)) | (tgt_cooords[:,0] > (self.r_max + coords_eps))
    #     out_of_range = out_of_range | (torch.abs(tgt_cooords[:,1]) > theta_max + coords_eps)
    #     out_of_range = out_of_range | (tgt_cooords[:,2] > self.phi_max + coords_eps)
    #     out_of_range = out_of_range | (tgt_cooords[:,2] < self.phi_min - coords_eps)
    #     valid_mask = ~out_of_range

    #     # discard non valid edges 
    #     tgt_cooords_val = tgt_cooords[valid_mask]
    #     i_val = self.i[valid_mask]
    #     j_val = self.j[valid_mask]

    #     valid_edges_num = i_val.shape[0]

    #     if valid_edges_num == 0: 
    #         return None, None, i_val, j_val, valid_mask

    #     # transform to fls values
    #     tgt_coords_val_fls = self.scale_phisical2fls(tgt_cooords_val)

    #     # --- get correlation for valid edges --- 

    #     # get grid to sample pixels from feature map 
    #     search_size = self.corr_neighbour + self.patch_size - 1

    #     r_range = torch.arange(-(search_size // 2), search_size // 2 + 1, device=device).float()
    #     dy, dx = torch.meshgrid(r_range, r_range, indexing="ij")
    #     offsets = torch.stack([dx, dy], dim=-1)

    #     center_coords_lv1 = tgt_coords_val_fls[:, [1, 0]].view(valid_edges_num, 1, 1, 2) 

    #     grid1 = center_coords_lv1 + offsets.unsqueeze(0)
    #     norm_factor1 = torch.tensor([(self.fls_w - 1) / 2.0, (self.fls_h - 1) / 2.0], device=device)
    #     grid1 = (grid1 / norm_factor1) - 1.0

    #     center_coords_lv2 = center_coords_lv1 / self.encoder_downsize

    #     grid2 = center_coords_lv2 + offsets.unsqueeze(0)
    #     norm_factor2 = torch.tensor([(self.fls_w // self.encoder_downsize - 1) / 2.0, (self.fls_h // self.encoder_downsize - 1) / 2.0], device=device)
    #     grid2 = (grid2 / norm_factor2) - 1.0

    #     # get features patches from fmaps
    #     b, n, c, h, w = self.fmap1.shape
    #     fmap1_cpy, fmap2_cpy = self.fmap1.view(b*n, c, h, w), self.fmap2.view(b*n, c, h//self.encoder_downsize, w//self.encoder_downsize)

    #     target_patches_fmap1 = F.grid_sample(fmap1_cpy[j_val], grid1, mode='bilinear', padding_mode='zeros', align_corners=True) #TODO:
    #     target_patches_fmap2 = F.grid_sample(fmap2_cpy[j_val], grid2, mode='bilinear', padding_mode='zeros', align_corners=True)

    #     target_patches_fmap1 = F.unfold(target_patches_fmap1, kernel_size=(self.patch_size, self.patch_size), stride=1)
    #     target_patches_fmap2 = F.unfold(target_patches_fmap2, kernel_size=(self.patch_size, self.patch_size), stride=1)

    #     target_patches_fmap1 = target_patches_fmap1.view(valid_edges_num, self.fmap_c, self.patch_size*self.patch_size, self.corr_neighbour*self.corr_neighbour)
    #     target_patches_fmap2 = target_patches_fmap2.view(valid_edges_num, self.fmap_c, self.patch_size*self.patch_size, self.corr_neighbour*self.corr_neighbour)

    #     # get patches 
    #     b, n, p, c1, d = self.patches_f.shape
    #     c2 = self.patches_c.shape[3]

    #     patches_f = self.patches_f.view(b*n*p, c1, d)
    #     patches_c = self.patches_c.view(b*n*p, c2)

    #     act_patches_f = patches_f[i_val, :, :]
    #     act_patches_c = patches_c[i_val, :]
        
    #     # calc correlation and connect to single tensor 
    #     corr_map1 = torch.einsum('ncpr, ncp -> nr', target_patches_fmap1, act_patches_f)
    #     corr_map2 = torch.einsum('ncpr, ncp -> nr', target_patches_fmap2, act_patches_f)

    #     corr_map = torch.cat((corr_map1.reshape(valid_edges_num, -1), corr_map2.reshape(valid_edges_num, -1)), dim=-1) 

    #     return corr_map, act_patches_c, i_val, j_val, valid_mask


    def get_hidden_state(self, valid_mask):
        return self.hidden_state[valid_mask, :]

    def update_hidden_state(self, h, valid_mask):
        self.hidden_state[valid_mask, :] = h

        
    def scale_fls2phisical(self, coords):

        # range r - measured by sonar
        r_norm = coords[:, 0] / self.fls_h
        r = r_norm * (self.r_max - self.r_min) + self.r_min

        # azimuth angle theta - measured by sonar
        theta_norm = coords[:, 1] / self.fls_w - 0.5
        theta = theta_norm * self.fov_horizontal 
        
        return torch.stack([r, theta], dim = 1)

    def scale_phisical2fls(self, coords):

        # range r - measured by sonar
        r_norm = (coords[:, 0] - self.r_min) / (self.r_max - self.r_min)
        r = r_norm * self.fls_h

        # azimuth angle theta - measured by sonar
        theta_norm = coords[:, 1] / self.fov_horizontal 
        theta = (theta_norm + 0.5) * self.fls_w
        
        return torch.stack([r, theta], dim = 1)
    
    def reset(self):
        '''Clear dynamic graph'''
        self.i = torch.empty(0, dtype=torch.long, device=self.i.device)
        self.j = torch.empty(0, dtype=torch.long, device=self.j.device)
        self.hidden_state = None
        self.frame_n = 0
