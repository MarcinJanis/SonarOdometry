import torch
import torch.nn as nn
import torch.nn.functional as F


class Update(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        # --- get model configuration ---
        self.fmap_c = model_cfg.FEATURES_OUTPUT_CH
        self.corr_neighbour = model_cfg.CORR_NEIGHBOUR
        self.patch_size = model_cfg.PATCH_SIZE 
        
        # correlation preprocess net
        corr_input_dim = self.fmap_c*self.corr_neighbour*self.corr_neighbour*self.patch_size*self.patch_size
        hidden_state_dim = model_cfg.CONTEXT_OUTPUT_CH

        self.corr_net = nn.Sequential(
            nn.Linear(corr_input_dim, hidden_state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.LayerNorm(hidden_state_dim, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state_dim, hidden_state_dim)
        )

        self.norm = nn.LayerNorm(hidden_state_dim, eps=1e-3)

    
    def forward(self, h, flow, corr, ctx, ii, jj, kk):
        
        '''
        Update operator
        
        :param h: hidden state 
        :param corr: correlation tensor
        :param ctx: context patches tensor
        :param flow: current correction of posses and weights 
        :param ii: buffer indexes of source frame for valid patches
        :param jj: buffer indexes of target frame for valid patches
        :param kk: gloabl ids of valid patches 
        '''

        corr = self.corr_net(corr)
        h = h + ctx + corr

        h = self.norm(h)




        pass

def neighbours(patch_idx, target_frame, range = 1):
    

    base = torch.stack([patch_idx, target_frame], dim=1) # shape (n, 2)
    prev = torch.stack([patch_idx, target_frame - range], dim=1)
    next = torch.stack([patch_idx, target_frame + range], dim=1)

    base = base.unsqueeze(0).permute(0, 2, 1) # shape  (1, 2, n)
    prev = prev.unsqueeze(-1) # shape (n, 2, 1)
    next = next.unsqueeze(-1) # shape (n, 2, 1)

    # search in past frames
    mask = (prev == base) # shape (n, 2, n)
    match = mask.all(dim=1)
    # if mask[i, :, k].all() == True, that means position i in prev (prev[i, :]) exist in base on position k (base[k, :]
    tgt_idx, base_idx_prev = match.nonzero(as_tuple=True) # each shape (t,) where t is number of matches and base[base_idx] matches with prev[tgt_idx]

    # search in future frames
    mask = (next == base) # shape (n, 2, n)
    match = mask.all(dim=1)
    # if mask[i, :, k].all() == True, that means position i in prev (prev[i, :]) exist in base on position k (base[k, :]
    tgt_idx, base_idx_next = match.nonzero(as_tuple=True) # each shape (t,) where t is number of matches and base[base_idx] matches with prev[tgt_idx]
    
    return base_idx_prev, base_idx_next
    
    

