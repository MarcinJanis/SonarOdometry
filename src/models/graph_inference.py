import torch
import torch.nn as nn
import torch.nn.functional as F

import math 
import numpy as np

from .patchifier import Patchifier
from .utils import project_points, transform_to_global


class Graph(nn.Module):
    def __init__(self, model_cfg, sonar_cfg):
        super().__init__()

        self.device = None

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
        self.buff_size = model_cfg.BUFF_SIZE 

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

        self.max_edges_num = 2 *  self.buff_size * self.patches_per_frame * self.time_window
        
        # --- init buffers ---

        # --- poses and time stamp buffers ---
        self.register_buffer('time', torch.zeros((self.buff_size), dtype=torch.float), persistent=False) # time stamp

        zero_poses = torch.zeros((self.buff_size, 7), dtype=torch.float)
        zero_poses[:, -1] = 1.0
        self.register_buffer('poses', zero_poses, persistent=False) # poses 
        
        # --- frame buffers ---
        self.register_buffer('fmap1', torch.zeros((self.buff_size, self.fmap_c, self.fmap_h, self.fmap_w), dtype = torch.float), persistent=False) # frames: features map
        self.register_buffer('fmap2', torch.zeros((self.buff_size, self.fmap_c, self.fmap_h // self.encoder_downsize, self.fmap_w // self.encoder_downsize), dtype = torch.float), persistent=False) # frames: features map 

        # --- patches buffers ---
        self.register_buffer('patches_f', torch.zeros((self.buff_size, self.patches_per_frame,  self.fmap_c, self.patch_size*self.patch_size), dtype = torch.float), persistent=False) # patches: features
        self.register_buffer('patches_c', torch.zeros((self.buff_size, self.patches_per_frame,  self.cmap_c), dtype = torch.float), persistent=False) # patches: context

        self.register_buffer('patch_coords', torch.zeros((self.buff_size, self.patches_per_frame, 3), dtype = torch.float), persistent=False) # points (r, theta, phi) refered to patches in real world units
        
        # --- dynamic edges tensors --- 
        self.i = torch.empty(0, dtype=torch.long)
        self.j = torch.empty(0, dtype=torch.long)
        self.hidden_state = torch.empty(0, dtype=torch.float)

        # 
        self.n = 0 # Global frames counter            

        # --- Modules ---
        self.patchifier = Patchifier(model_cfg)

        
    def _add_patches(self, patch_coords, patches_f, patches_c):
        local_n = self.g2l_frame_idx(self.n)

        # pop old patches data if buffer full
        patch_coords_poped, idx_poped = None, None
        if self.n >= self.buff_size:
            patch_coords_poped = self.patch_coords[local_n, :, :]
            idx_poped = np.arange((self.n - self.buff_size)*self.patches_per_frame, 
                                  (self.n + 1 - self.buff_size)*self.patches_per_frame)

        # add new patches data to buffers
        self.patch_coords[local_n, :, :] = patch_coords
        self.patches_f[local_n, :, :, :] = patches_f
        self.patches_c[local_n, :, :] = patches_c

        return idx_poped, patch_coords_poped

    def _add_frame(self, fmap):
        b, n, c, h, w = fmap.shape
        local_n = self.g2l_frame_idx(self.n)

        fmap_downsize = F.avg_pool2d(fmap.view(b*n, c, h, w), self.encoder_downsize, self.encoder_downsize)

        self.fmap1[local_n, :, :, :] = fmap.view(c, h, w)
        self.fmap2[local_n, :, :, :] = fmap_downsize.view(c, h // self.encoder_downsize, w // self.encoder_downsize)
        
        
    def _add_pose(self, pose, time):
        local_n = self.g2l_frame_idx(self.n)

        # pop old data if buffer full
        pose_poped, time_poped, idx_poped = None, None, None
        if self.n >= self.buff_size:
            pose_poped = self.poses[local_n, :]
            time_poped = self.time[local_n]
            idx_poped = self.n - self.buff_size

        self.poses[local_n, :] = pose
        self.time[local_n] = time

        return idx_poped, pose_poped, time_poped
       

    # global to local frame idx
    def g2l_frame_idx(self, idx):
        return idx % self.buff_size
    
    # global to local patch idx
    def g2l_patch_idx(self, idx):
        frame_idx = (idx // self.patches_per_frame) % self.buff_size
        patch_number = idx % self.patches_per_frame
        return frame_idx, patch_number

    def extract_features(self, frames, pose, time):

        # add new data to global counter
        self.n += 1

        # extract features
        coords, patches_f, patches_c, fmap = self.patchifier(frames) 
        
        # add new patches coords
        b, n, p, d = coords.shape
        coords_r_theta = self.scale_fls2phisical(coords.view(b*n*p, 2)) # coords of patches (r, theta)  
        coords_r_theta = coords_r_theta.view(b*n, p, 2)
        coords_phi = self.init_phi(b*n, p, self.device)
        patch_coords = torch.cat([coords_r_theta, coords_phi], dim=-1)
        patch_idx, patch_coords_poped = self._add_patches(patch_coords, patches_f, patches_c)

        # add new frames feature maps
        self._add_frame(fmap)

        # add new pose
        frame_idx, pose_poped, time_poped = self._add_pose(pose, time)

        # return poped items
        return (frame_idx, pose_poped, time_poped, patch_idx, patch_coords_poped)


    def create_edges(self):

        new_i, new_j = [], [] 
        
        start_tgt_frame = max(0, self.n - self.time_window)
        for tf in range(start_tgt_frame, self.n): # for each target frame in range time window      
                # edges: new patches -> old frames
                new_i.append(torch.arange(self.n * self.patches_per_frame, (self.n + 1) * self.patches_per_frame, device=self.device)) 
                new_j.append(torch.full((self.patches_per_frame,), tf, device=self.device))
                
                # edges: old patches -> new frame
                new_i.append(torch.arange(tf * self.patches_per_frame, (tf + 1) * self.patches_per_frame, device=self.device))
                new_j.append(torch.full((self.patches_per_frame,), self.n, device=self.device))

        if len(new_i) == 0:
            return # if there is no new edges
        
        new_i = torch.cat(new_i, dim=0)
        new_j = torch.cat(new_j, dim=0)
        new_edges = new_i.shape[0]
        new_hidden_state = torch.zeros((new_edges, self.cmap_c), device=self.device, dtype=torch.float)

        # add to existing edges
        self.i = torch.cat([self.i, new_i], dim=0)
        self.j = torch.cat([self.j, new_j], dim=0)
        self.hidden_state = torch.cat([self.hidden_state, new_hidden_state], dim=0)

        # delete obsolete edges
        # print(f'edges1\ni: {self.i}\nj: {self.j}')
        edges_to_keep = (self.j >= (self.n - self.buff_size)) # delete edges that points to obsolete frames
        # edges_to_keep = ~edges_to_del

        self.i = self.i[edges_to_keep]
        self.j = self.j[edges_to_keep] 
        # print(f'edges2\ni: {self.i}\nj: {self.j}')
        self.hidden_state = self.hidden_state[edges_to_keep]
        
        # edges_lim = max(0, self.i.shape[0] - self.max_edges_num) 
        # if edges_lim > 0:
        #     self.i = self.i[edges_lim:]
        #     self.j = self.j[edges_lim:] 
        #     self.hidden_state = self.hidden_state[edges_lim:]

    def corr(self, coords_eps, device):

        # --- reproject points --- 

        # src and tgt framem idxs
        # print('i',self.i)
        src_frame_idx, src_patch_idx = self.g2l_patch_idx(self.i)

        tgt_frame_idx = self.g2l_frame_idx(self.j)
        # print('src_frame_idx',src_frame_idx)
        # print('src_patch_idx',src_patch_idx)
        src_poses = self.poses[src_frame_idx, :]
        tgt_poses = self.poses[tgt_frame_idx, :]

        src_coords = self.patch_coords[src_frame_idx, src_patch_idx, :]

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

        # --- get correlation for valid edges --- 

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
        grid1 = (grid1 / norm_factor1) - 1.0

        # downsample target coords with offsets
        center_coords_lv2 = center_coords_lv1 / self.encoder_downsize

        # create downsample sampling grid
        grid2 = center_coords_lv2 + offsets.unsqueeze(0)
        norm_factor2 = torch.tensor([(self.fls_w // self.encoder_downsize - 1) / 2.0, (self.fls_h // self.encoder_downsize - 1) / 2.0], device=device)
        grid2 = (grid2 / norm_factor2) - 1.0

        # get features patches from fmaps

        j_val_local = self.g2l_frame_idx(j_val)

        fmap1_expand = self.fmap1[j_val_local, :, :, :]
        fmap2_expand = self.fmap2[j_val_local, :, :, :]

        target_patches_fmap1 = F.grid_sample(fmap1_expand, grid1, mode='bilinear', padding_mode='zeros', align_corners=True)
        target_patches_fmap2 = F.grid_sample(fmap2_expand, grid2, mode='bilinear', padding_mode='zeros', align_corners=True)

        target_patches_fmap1 = target_patches_fmap1.view(1, valid_edges_num * self.fmap_c, search_size, search_size)
        target_patches_fmap2 = target_patches_fmap2.view(1, valid_edges_num * self.fmap_c, search_size, search_size)

        # get patches features 
        i_val_frame_local, i_val_patch_local = self.g2l_patch_idx(i_val)
        act_patches_f = self.patches_f[i_val_frame_local, i_val_patch_local, :, :]
        act_patches_c = self.patches_c[i_val_frame_local, i_val_patch_local, :]
        
        n, c1, d = act_patches_f.shape
        patch_features_kernel = act_patches_f.view(n, c1, self.patch_size, self.patch_size) # patches features kernel

        # perform conv2d with group = valid_edges_num
        # each of kernels, represnting each patch have acces to features of corresponding fmap features
        corr_map1 = F.conv2d(target_patches_fmap1, patch_features_kernel, groups=valid_edges_num)
        corr_map2 = F.conv2d(target_patches_fmap2, patch_features_kernel, groups=valid_edges_num)
        # output shape: (1, valid_edges_num, corr_neighbour, corr_neighbour)


        corr_map = torch.cat((corr_map1.view(valid_edges_num, -1), corr_map2.view(valid_edges_num, -1)), dim=-1) 

        return corr_map, act_patches_c, i_val, j_val, valid_mask
        
        
        # target_patches_fmap1 = F.unfold(target_patches_fmap1, kernel_size=(self.patch_size, self.patch_size), stride=1)
        # target_patches_fmap2 = F.unfold(target_patches_fmap2, kernel_size=(self.patch_size, self.patch_size), stride=1)

        # target_patches_fmap1 = target_patches_fmap1.view(valid_edges_num, self.fmap_c, self.patch_size*self.patch_size, self.corr_neighbour*self.corr_neighbour)
        # target_patches_fmap2 = target_patches_fmap2.view(valid_edges_num, self.fmap_c, self.patch_size*self.patch_size, self.corr_neighbour*self.corr_neighbour)

        # # get patches features 
        # i_val_frame_local, i_val_patch_local = self.g2l_patch_idx(i_val)
        # act_patches_f = self.patches_f[i_val_frame_local, i_val_patch_local, :, :]
        # act_patches_c = self.patches_c[i_val_frame_local, i_val_patch_local, :]
        
        # # calc correlation and connect to single tensor 
        # corr_map1 = torch.einsum('ncpr, ncp -> nr', target_patches_fmap1, act_patches_f)
        # corr_map2 = torch.einsum('ncpr, ncp -> nr', target_patches_fmap2, act_patches_f)

        # corr_map = torch.cat((corr_map1.reshape(valid_edges_num, -1), corr_map2.reshape(valid_edges_num, -1)), dim=-1) 

        # return corr_map, act_patches_c, i_val, j_val, valid_mask

    def get_last_poses(self, num=2):
        local_n = self.g2l_frame_idx(self.n)
        if num == 1:
            x = self.poses[local_n, :].unsqueeze(0)
            t = self.time[local_n].unsqueeze(0)
        else:
            x = [self.poses[local_n - i, :].unsqueeze(0) for i in range(num) ]
            t = [self.time[local_n - i].unsqueeze(0) for i in range(num) ]
        return x, t

    def get_poses(self):
        return self.poses
    
    def get_patch_coords(self):
        b, p, _ = self.patch_coords.shape
        coords_r_theta = self.patch_coords[:, :, :2]
        coords_phi = self.patch_coords[:, :, 2:3]
        return coords_r_theta, coords_phi

    def update_poses(self, poses):
        self.poses[:, :] = poses

    def update_patch_coords(self, phi):
        self.patch_coords[:, :, 2] = phi.squeeze(-1)
    
    def get_hidden_state(self, valid_mask):
        return self.hidden_state[valid_mask, :]

    def update_hidden_state(self, h, valid_mask):
        self.hidden_state[valid_mask, :] = h

    def init_phi(self, n, p, device, mode = 'rand'):
        if self.phi_init_mode == 'rand':
            coords_phi = torch.rand((n, p, 1), device=device, dtype=torch.float) * (self.phi_max - self.phi_min) + self.phi_min
        else: 
            coords_phi = torch.zeros((n, p, 1), device=device, dtype=torch.float) # init elevation angle with zeros
        return coords_phi

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
        self.n = 0