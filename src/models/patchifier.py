import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import Encoder

# for debug purpose:
import numpy as np 
import cv2

class Patchifier(nn.Module):
    def __init__(self, patches_per_frame, patch_size:int=3, grid_size:tuple=(24, 24)):
        super().__init__()

        self.debug_dict = {}

        assert patches_per_frame <= grid_size[0]*grid_size[1], f'[Error]: Patchifier module.\n number of patches can\'t be greater than number of cells in grid.'

        self.patches_per_frame = patches_per_frame
        self.patch_size = patch_size

        self.grid_size_h, self.grid_size_w = grid_size

        self.feature_extractor = Encoder(in_ch = 1, out_ch = 128, dim = 32, dropout=0.5, norm_fn='instance')
        self.context_extractor = Encoder(in_ch = 1, out_ch = 128, dim = 32, dropout=0.5, norm_fn=None)
        self.downsize_factor = 4

    def _norm_frame(self, frame, v_max = 255):
        # mean = torch.mean(frame)
        # std = torch.std(frame)
        # return (frame - mean)/std * v_max
        # t_min, t_max = frame.min(), frame.max()
        # return (frame - t_min) / (t_max - t_min + 1e-8)
        pass
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
        coords = torch.stack([x_best, y_best], dim=-1).float() / self.downsize_factor

        return coords # shape (bn, patches_per_frame, 2)
    
    def _extract_patches(self, coords, map, patch_size, stride = 1):

        device = map.device

        bn, c, h, w = map.shape

        # (b, n, c2, h2, w2) -> (b, n, c_2, num_patches_h, num_patches_w, patch_size, patch_size)``
        map = map.unfold(-1, patch_size, stride).unfold(-2, patch_size, stride)

        # b, n, c_2, num_patches_h, num_patches_w, _, _ = map.shape
        iy = ((coords[..., 1] - (patch_size // 2)) // stride).long().clamp(0, h - 1)
        ix = ((coords[..., 0] - (patch_size // 2)) // stride).long().clamp(0, w - 1)
    
        batch_idx = torch.arange(bn, device)

        patches = map[batch_idx,:,:,ix,iy,:]
        return patches # (bn, c_2, num_patches_h, num_patches_w, patch_size, patch_size)``

    def _patchifier_debug(self, frame, coords):
            
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

            for i in range(self.grid_size_h + 1): 
                y = int(i * step_y)
                cv2.line(frame_np, (0, y), (w, y), (255, 0, 0), 1)

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
        cmap = self.context_extractor(frame)

        b, n, c, h, w = fmap.shape
        bn = b*n

        if mode == 'harris':
            # get strongest structures from frame
            g = self._harris_response(frame, ksize=7, padding=3)
            coords = self._get_best_coords(g)






        # for debug purpose:
        elif mode == 'harris_debug':
            
            # normalize frame
            # frame = self._norm_frame(frame, v_max=255)
            # get strongest structures from frame
            g = self._harris_response(frame, ksize=7, padding=3)
            # get coords
            coords = self._get_best_coords(g)
            self._patchifier_debug(frame, coords)
            
        return g
        

        