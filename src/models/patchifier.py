import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import Encoder

# for debug purpose:
import numpy as np 
import cv2

class Patchifier(nn.Module):
    # def __init__(self, patches_per_frame, patch_size:int=3, grid_size:tuple=(24, 24), debug_mode = False):
    def __init__(self, cfg, debug_mode = False):
        super().__init__()
        
        self.cfg = cfg

        self.debug_mode = debug_mode
        self.debug_dict = {}

        # --- Import configuration 
        self.patches_per_frame = self.cfg.PATCHES_PER_FRAME
        self.patch_size = self.cfg.PATCH_SIZE

        self.grid_size_h, self.grid_size_w = self.cfg.PATCHES_GRID_SIZE.x, self.cfg.PATCHES_GRID_SIZE.y

        
        self.feature_extractor = Encoder(in_ch = 1, 
                                         out_ch = self.cfg.FEATURES_OUTPUT_CH,
                                         dim = self.cfg.FEATURES_MAP_FIRST_DIM, 
                                         dropout=0.5, 
                                         norm_fn=self.cfg.ENCODER_NORM_METHOD)
        
        self.context_extractor = Encoder(in_ch = 1, 
                                         out_ch = self.cfg.FEATURES_OUTPUT_CH,
                                         dim = self.cfg.FEATURES_MAP_FIRST_DIM, 
                                         dropout=0.5, 
                                         norm_fn=None)
        
        self.downsize_factor = self.cfg.ENCODER_DOWNSIZE

        assert self.patches_per_frame <= self.grid_size_h*self.grid_size_w , f'[Error]: Patchifier module.\n number of patches can\'t be greater than number of cells in grid.'


    def _harris_response(self, frame, ksize=7, padding=3):

        b, n, c, h, w = frame.shape
        frame = frame.view(b*n, c, h, w)

        # Gradients x and y
        dx = F.pad(frame[:,:,:,1:] - frame[:,:,:,:-1], (0,1,0,0), mode='reflect')
        dy = F.pad(frame[:,:,1:,:] - frame[:,:,:-1,:], (0,0,0,1), mode='reflect')
        
        # Structural matrix components
        Ixx = F.avg_pool2d(dx**2, ksize, stride=1, padding=padding)
        Iyy = F.avg_pool2d(dy**2, ksize, stride=1, padding=padding)
        Ixy = F.avg_pool2d(dx*dy, ksize, stride=1, padding=padding)
        
        # Shi-Tomasi response (min eigenvalue form)
        # determinant = Ixx*Iyy - Ixy**2, trace = Ixx + Iyy
        trace = Ixx + Iyy
        determinant = Ixx * Iyy - Ixy**2
        
        response = determinant / (trace + 1e-8)

        if self.debug_mode:
            self.debug_dict['harris_response'] =  response.detach().cpu().squeeze(0).squeeze(0).numpy().astype(np.uint8)

        return response

    def _get_best_coords(self, g):

        device = g.device

        bn, c, h, w = g.shape

        pix_per_cell_h = (h + self.grid_size_h - 1) // self.grid_size_h
        pix_per_cell_w = (w + self.grid_size_w - 1) // self.grid_size_w

        pad_h = pix_per_cell_h * self.grid_size_h - h 
        pad_w = pix_per_cell_w * self.grid_size_w - w
        
        if pad_h > 0 or pad_w > 0:
            g = F.pad(g, (0, pad_w, 0, pad_h), mode='constant', value=0) # pad: right side and bottom

        # devide into grid of cells
        g = g.view(bn, 1, self.grid_size_h, pix_per_cell_h, self.grid_size_w, pix_per_cell_w)
        g = g.permute(0, 2, 4, 1, 3, 5).contiguous()
        g = g.view(bn, self.grid_size_h, self.grid_size_w, pix_per_cell_h * pix_per_cell_w)

        # find strongest features in each cell
        max_vals, max_idx = torch.max(g, dim=-1) # max_idx.shape = (bn, self.grid_size_h, self.grid_size_w), value -> idx of max value in cell

        # local (cell) index -> global index
        y_local = torch.div(max_idx, pix_per_cell_w, rounding_mode='floor')
        x_local = max_idx % pix_per_cell_w

        row_idx = torch.arange(self.grid_size_h, device=device)
        col_idx = torch.arange(self.grid_size_w, device=device)   

        y_offset = (pix_per_cell_h * row_idx).view(1, self.grid_size_h, 1)
        x_offset = (pix_per_cell_w * col_idx).view(1, 1, self.grid_size_w)

        y_global = y_local + y_offset
        x_global = x_local + x_offset

        # get patches_per_frame best results 
        vals_flat = max_vals.view(bn, -1)
        y_flat = y_global.view(bn, -1)
        x_flat = x_global.view(bn, -1)
        
        _, top_idxs = torch.topk(vals_flat, self.patches_per_frame, dim=-1, largest=True, sorted=True) 

        y_best = torch.gather(y_flat, 1, top_idxs)
        x_best = torch.gather(x_flat, 1, top_idxs)
        
        # stack coords, rescale to downsized shape (compliance with output of encoder)
        coords = torch.stack([x_best, y_best], dim=-1).float()
        coords = coords.view(bn, self.patches_per_frame, 2)
        return coords

    def _get_patches(self, coords, fmap, cmap):

        device = fmap.device
        # scale coords to features map
        coords = coords / self.downsize_factor 

        # offsets to get patches
        r = torch.arange(-(self.patch_size//2), self.patch_size//2 + 1, device=device)
        dy, dx = torch.meshgrid(r, r, indexing="ij")
        coords_offsets = torch.stack([dx, dy], dim=-1).float() # shape [K, K, 2]

        # add offsets dim to coords
        coords_p = coords.unsqueeze(-2).unsqueeze(-2) + coords_offsets.unsqueeze(0).unsqueeze(0) # [B*N, patches_per_frame, K, K, 2]
        
        # normalize to (-1, 1) range
        bn, c, h, w = fmap.shape
        
        xp_norm = (2 * coords_p[:, :, :, :, 0] + 1) / w - 1
        yp_norm = (2 * coords_p[:, :, :, :, 1] + 1) / h - 1

        # sampling grid with norm coords of patches ceneter 
        grid_p = torch.stack([xp_norm, yp_norm], dim=-1) # grid shape [b*n, patcher_per_frame, K, K 2]

        x1_norm = (2 * coords[:, :, 0] + 1) / w - 1
        y1_norm = (2 * coords[:, :, 1] + 1) / h - 1
        grid_1 = torch.stack([x1_norm, y1_norm], dim=-1)

        # sample patches
        patches_f = torch.nn.functional.grid_sample(
            fmap.view(bn, c, h, w),
            grid_p.view(bn, self.patches_per_frame*self.patch_size*self.patch_size, 1, 2), # shape: [frames_num, total_pts_num, 1, xy]
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        patches_c = torch.nn.functional.grid_sample(
            cmap.view(bn, c, h, w),
            grid_1.view(bn, self.patches_per_frame, 1, 2), # shape: [frames_num, total_pts_num, 1, xy]
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )
        
        patches_f = patches_f.view(bn, c, self.patches_per_frame, self.patch_size * self.patch_size)
        patches_f = patches_f.permute(0, 2, 1, 3)
        # patches_f = patches_f.view(bn, self.patches_per_frame, c, self.patch_size*self.patch_size)
        
        patches_c = patches_c.view(bn, c, self.patches_per_frame)
        patches_c = patches_c.permute(0, 2, 1)
        # patches_c = patches_c.view(bn, self.patches_per_frame, c)
        
        return patches_f, patches_c

    def _patchifier_draw_keypoints(self, frame, coords):
            
            # create copy of new frame
            frame_np = frame.squeeze(0).squeeze(0).squeeze(0)
            frame_np = frame_np.detach().cpu().numpy()
            frame_np = frame_np.astype(np.uint8)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)

            # create copy of coords
            coords_np = coords.squeeze(0).detach().cpu().numpy()
            coords_np = coords_np.astype(np.int16) * self.downsize_factor  

            # draw grid 
            h, w, _ = frame_np.shape

            step_y = h // self.grid_size_h
            step_x = w // self.grid_size_w

            # draw verticall lines
            for i in range(self.grid_size_h + 1): 
                y = int(i * step_y)
                cv2.line(frame_np, (0, y), (w, y), (255, 0, 0), 1)

            # draw horizontal lines
            for i in range(self.grid_size_w + 1):
                x = int(i * step_x)
                cv2.line(frame_np, (x, 0), (x, h), (255, 0, 0), 1)

            # draw points
            for i in range(coords_np.shape[0]):
                x, y = coords_np[i,:]     
                cv2.circle(frame_np, (x, y), 2, (0, 255, 0), 4)

            # save in debug dict
            self.debug_dict['key_points'] =  frame_np
            
            

    # extract patches from new frame
    def forward(self, frame, mode = 'harris'):

        # frame shape: (b, n, c, h, w)
        # b - batch size
        # n - frames in series 
        # c - channels
        # h 
        # w

        fmap = self.feature_extractor(frame)
        imap = self.context_extractor(frame)

        # print(f'fmap shape: {fmap.shape}')
        bn, c, h, w = fmap.shape
        # bn = b*n
        
        if mode == 'harris':
            # get strongest structures from frame
            g = self._harris_response(frame, ksize=7, padding=3) # g.shape = [b*n, c, h, w]
            
            # get coords
            coords = self._get_best_coords(g) # coords.shape = [b*n, self.patches_per_frame, 2], coords are in orginal frame coords system
            patches_f, patches_c = self._get_patches(coords, fmap, imap) #patches.shape = [b*n, self.patches_per_frame, c, self.patch_size, self.patch_size]

            # debug functionalities
            if self.debug_mode:
                # draw grid on orginal frame, mark key points, accesible via self.debug_dict['key_points'], as numpy RGB img
                self._patchifier_draw_keypoints(frame, coords) 

        
        return coords, patches_f, patches_c, fmap
        

        
