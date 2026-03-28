import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

from .encoders import Encoder

import numpy as np 
import cv2

class Patchifier(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg

        # --- Import configuration ---
        self.patches_per_frame = self.cfg.PATCHES_PER_FRAME
        self.patch_size = self.cfg.PATCH_SIZE
        self.grid_size_h, self.grid_size_w = self.cfg.PATCHES_GRID_SIZE.x, self.cfg.PATCHES_GRID_SIZE.y
        self.downsize_factor = self.cfg.ENCODER_DOWNSIZE
        
        self.feature_extractor = Encoder(in_ch = 1, 
                                         out_ch = self.cfg.FEATURES_OUTPUT_CH,
                                         dim = self.cfg.FEATURES_MAP_FIRST_DIM, 
                                         dropout=0.5, 
                                         norm_fn=self.cfg.ENCODER_NORM_METHOD,
                                         downsize=self.downsize_factor)
        
        self.context_extractor = Encoder(in_ch = 1, 
                                         out_ch = self.cfg.CONTEXT_OUTPUT_CH,
                                         dim = self.cfg.CONTEXT_MAP_FIRST_DIM, 
                                         dropout=0.5, 
                                         norm_fn=None,
                                         downsize=self.downsize_factor)
        
        
        assert self.patches_per_frame <= self.grid_size_h*self.grid_size_w , f'[Error]: Patchifier module.\n number of patches can\'t be greater than number of cells in grid.'

    def _harris_response(self, frame, ksize=7, padding=3):
     
        # connect batch size and frames in series dimension
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

    def _hessian_det(self, frame, ksize=7, sigma=(0.5, 0.5)):
     
        # connect batch size and frames in series dimension
        b, n, c, h, w = frame.shape
        frame = frame.view(b*n, c, h, w)

        # Gaussian Blur
        blur = GaussianBlur(kernel_size=ksize, sigma=sigma)
        frame = blur(frame)
        
        # Gradients x and y
        dx = F.pad(frame[:,:,:,1:] - frame[:,:,:,:-1], (0,1,0,0), mode='reflect')
        dy = F.pad(frame[:,:,1:,:] - frame[:,:,:-1,:], (0,0,0,1), mode='reflect')

        # Second derivative 
        d2x = F.pad(dx[:,:,:,1:] - dx[:,:,:,:-1], (0,1,0,0), mode='reflect')
        d2y = F.pad(dy[:,:,1:,:] - dy[:,:,:-1,:], (0,0,0,1), mode='reflect')
        dxy = F.pad(dx[:,:,1:,:] - dx[:,:,:-1,], (0,0,0,1), mode='reflect')
        
        hessian_det = d2x * d2y - dxy**2

        return hessian_det

    
    def _DoG(self, frame, kernel_size = 5, sigma1 = (0.1, 2.0), sigma2 = (0.2, 1.0)):

        b, n, c, h, w = frame.shape
        frame = frame.view(b*n, c, h, w)
        
        blur1 = GaussianBlur(kernel_size=kernel_size, sigma=sigma1)
        blur2 = GaussianBlur(kernel_size=kernel_size, sigma=sigma2)

        img1 = blur1(frame)
        img2 = blur2(frame)
        
        dog = torch.clip(img1-img2, min=0.0, max=255.0)
                         
        return dog
                         
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
        coords = torch.stack([y_best, x_best], dim=-1).float()
        coords = coords.view(bn, self.patches_per_frame, 2)
        return coords

    def _get_patches(self, coords, fmap, cmap):

        device = fmap.device
        # scale coords to features map
        coords = coords / self.downsize_factor 

        # offsets to get patches
        r = torch.arange(self.patch_size, device=device).float() - (self.patch_size // 2)
        dy, dx = torch.meshgrid(r, r, indexing="ij")
        coords_offsets = torch.stack([dy, dx], dim=-1).float() # shape [K, K, 2]

        # add offsets dim to coords
        coords = coords.unsqueeze(-2)
        coords_p = coords.unsqueeze(-2) + coords_offsets.unsqueeze(0).unsqueeze(0) # [B*N, patches_per_frame, K, K, 2]
        
        # normalize to (-1, 1) range
        bn, c1, h, w = fmap.shape
        c2 = cmap.shape[1]

        yp_norm = (2 * coords_p[:, :, :, :, 0] + 1) / h - 1
        xp_norm = (2 * coords_p[:, :, :, :, 1] + 1) / w - 1

        # sampling grid with norm coords of patches ceneter 
        grid_p = torch.stack([yp_norm, xp_norm], dim=-1) # grid shape [b*n, patcher_per_frame, K, K 2]

        y1_norm = (2 * coords[:, :, :, 0] + 1) / h - 1
        x1_norm = (2 * coords[:, :, :, 1] + 1) / w - 1
        grid_1 = torch.stack([y1_norm, x1_norm], dim=-1)

        # sample patches
        patches_f = torch.nn.functional.grid_sample(
            fmap.view(bn, c1, h, w),
            grid_p.view(bn, self.patches_per_frame*self.patch_size*self.patch_size, 1, 2), # shape: [frames_num, total_pts_num, 1, xy]
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        patches_c = torch.nn.functional.grid_sample(
            cmap.view(bn, c2, h, w),
            grid_1.view(bn, self.patches_per_frame, 1, 2), # shape: [frames_num, total_pts_num, 1, xy]
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        patches_f = patches_f.view(bn, c1, self.patches_per_frame, self.patch_size * self.patch_size)
        patches_f = patches_f.permute(0, 2, 1, 3).contiguous()
        # patches_f = patches_f.view(bn, self.patches_per_frame, c, self.patch_size*self.patch_size)
        
        patches_c = patches_c.view(bn, c2, self.patches_per_frame)
        patches_c = patches_c.permute(0, 2, 1).contiguous()
        # patches_c = patches_c.view(bn, self.patches_per_frame, c)
        
        return patches_f, patches_c

    def get_visu(self, frames, coords, batch=0, frame_num=0):
         
            single_frame = frames[batch, frame_num, ...]
            single_coord = coords[batch, frame_num, ...]
          
            # create copy of new frame
            frame_np = single_frame.clone().detach().squeeze(0).cpu().numpy() 

            frame_np = (frame_np*255).astype(np.uint8) 
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)


            # create copy of coords
            coords_np = single_coord.detach().cpu().numpy()
            coords_np = coords_np.astype(np.int16)

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
                y, x = coords_np[i,:]     
                # print(f'pt: {i}: x={x}, y={y}')
                cv2.circle(frame_np, (x, y), 2, (0, 255, 0), 4)

            harris_response = self._harris_response(single_frame.unsqueeze(0).unsqueeze(0))
            harris_response = harris_response.clone().detach().squeeze(0).squeeze(0)
            harris_response = (harris_response*255).cpu().numpy().astype(np.uint8)
            
            return frame_np, harris_response
            
            

    # extract patches from new frame
    def forward(self, frame, mode = 'harris'):

        fmap = self.feature_extractor(frame)
        imap = self.context_extractor(frame)

        b, n, c1, h, w = fmap.shape
        c2 = imap.shape[2]

        # get strongest structures from frame
        if mode == 'harris':
            g = self._harris_response(frame, ksize=7, padding=3) # g.shape = [b*n, c, h, w]
        elif mode == 'DoG':
            g =  self._DoG(frame, kernel_size=5, sigma1=(0.1, 2.0), sigma2=(0.2, 1.0)) # g.shape = [b*n, c, h, w]
        elif mode == 'hessian':
            g = self._hessian_det(frame, ksize=7, sigma=(0.5, 0.5)) # g.shape = [b*n, c, h, w]
        else: 
            g = frame.view(b*n, c1, h, w)  # g.shape = [b*n, c, h, w]
            
        # get coords
        coords = self._get_best_coords(g) # coords.shape = [b*n, self.patches_per_frame, 2], coords are in orginal frame coords system
        patches_f, patches_c = self._get_patches(coords, fmap.view(b*n, c1 ,h, w), imap.view(b*n, c2 ,h, w)) #patches.shape = [b*n, self.patches_per_frame, c, self.patch_size, self.patch_size]

        # append dimesion for sepearate batch dimension
    
        patches_f = patches_f.view(b, n, self.patches_per_frame, c1, self.patch_size * self.patch_size)
        patches_c = patches_c.view(b, n, self.patches_per_frame, c2)
        coords = coords.view(b, n, self.patches_per_frame, 2)

        return coords, patches_f, patches_c, fmap
        
