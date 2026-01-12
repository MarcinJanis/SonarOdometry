import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import Encoder

class Patchifier(nn.Module):
    def __init__(self, patches_per_frame, patch_size:int=3, grid_size:tuple=(24, 24)):
        super().__init__()

        self.patches_per_frame = patches_per_frame
        self.patch_size = patch_size

        self.grid_size_h, self.grid_size_w = grid_size

        self.feature_extractor = Encoder(in_ch = 1, out_ch = 128, dim = 32, dropout=0.5)
        self.context_extractor = Encoder(in_ch = 1, out_ch = 128, dim = 32, dropout=0.5)
        self.downsize_factor = 4

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

        bn, h, w = g.shape

        pix_per_cell_h = (h + self.grid_size_h) // self.grid_size_h - 1
        pix_per_cell_w = (w + self.grid_size_w) // self.grid_size_w - 1

        pad_h = pix_per_cell_h * self.grid_size_h - h 
        pad_w = pix_per_cell_w * self.grid_size_w - w

        if pad_h > 0 or pad_w > 0:
            g = F.pad(g, (0, pad_w, 0, pad_h), mode='constant', value=0) # pad: right side and bottom

        # devide into grid of cells
        g = g.view(bn, 1, self.grid_size_h, pix_per_cell_h, self.grid_size_w, pix_per_cell_w)
        g = g.permute(0, 2, 4, 3, 5).contiguous()
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

        return coords
    
    def _get_patches(self, coords):
        pass

    # extract patches from new frame
    def forward(self, frame:torch.tensor, n:int, mode = 'harris'):

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
            # get coords
            coords = self._get_best_coords(g)


        