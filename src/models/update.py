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
        
        # --- update hidden state with new data --- 
        corr = self.corr_net(corr) # process correlation tensor
        h = h + ctx + corr # add to hidden state
        h = self.norm(h) # normalize





        pass



def neighbours(patch_idx, act_target_frame, range = 1):
    # dopracować!!!!
    base = torch.stack([patch_idx, act_target_frame], dim = 1) # shape (n, 2); [:,0] - patch idx, [:,1] - target frame

    # define searching targets
    prev = torch.stack([patch_idx, act_target_frame - range], dim = 1)
    next = torch.stack([patch_idx, act_target_frame - range], dim = 1)

    # search
    past_results =  torch.nonzero((base == prev))
    future_results =  torch.nonzero((base == next))

    past_results =  past_results.all(dim=1)
    future_results =  future_results.all(dim=1)

    past_idx = torch.nonzero(past_results, as_tuple=True)[0]
    future_idx = torch.nonzero(past_results, as_tuple=True)[0]

    return past_idx, future_idx
   