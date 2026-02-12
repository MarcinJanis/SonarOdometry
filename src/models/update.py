import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


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

        self.corr_net = nn.Sequential((
            nn.Linear(corr_input_dim, hidden_state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.LayerNorm(hidden_state_dim, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state_dim, hidden_state_dim)
        ))

        self.norm = nn.LayerNorm(hidden_state_dim, eps=1e-3)

        self.c1 = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state_dim, hidden_state_dim))

        self.c2 = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state_dim, hidden_state_dim)

        self.patches_agg = SoftAgg(dim=hidden_state_dim, expand=True)
        self.edges_agg = SoftAgg(dim=hidden_state_dim, expand=True)

        # recurrent net here:
        self.gru = nn.Sequential(
            nn.LayerNorm(hidden_state_dim, eps=1e-3),
            GatedResidual(hidden_state_dim), 
            nn.LayerNorm(hidden_state_dim, eps=1e-3),
            GatedResidual(hidden_state_dim)
        )


    
    def forward(self, h, flow, corr, ctx, source_frame_ids, target_frames_ids, patches_ids):
        
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

        corr = self.corr_net(corr) # make sure that hidden state and rest have shape (1, n, 1), where 1 is batch size, n - edges number and 1 id if its necessey, maybe not 
        h = h + ctx + corr

        h = self.norm(h)

        # for each edge find edge idx, where same patch is matched with previous or next target frame (in time) 
        prev_idx, next_idx = neighbours(patch_idx, target_frame, device, range = 1)

        prev_mask = (prev_idx >= 1).float.reshape(1, -1, 1)
        next_mask = (next_idx >= 1).float.reshape(1, -1, 1)

        h = h + self.c1(prev_mask * h[:, prev_idx]) # add to hidden state information about temporal patches neighbours 
        h = h + self.c2(next_mask * h[:, next_idx]) # add to hidden state information about temporal patches neighbours 

        h = h + self.patches_agg(h, patches_ids)
        h = h + self.edges_agg(h, source_frame_ids*12345 + =target_frames_ids)


        
        return h, 



class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid())

        self.res = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.gate(x) * self.res(x)

class SoftAgg(nn.Module):
    def __init__(self, dim=512, expand=True):
        super(SoftAgg, self).__init__()
        self.dim = dim
        self.expand = expand
        self.linear1 = nn.Linear(self.dim, self.dim)
        self.linear2 = nn.Linear(self.dim, self.dim)
        self.linaer3 = nn.Linear(self.dim, self.dim)

    def forward(self, x, id):

        # assign each element of id tensor to group, a unique group assigment is achived
        # unique, idx = torch.unique(input, return_inverse=True) # returns: unique - unique values that occurs in input tensor, idx - for each element idx of value from unique is assigned 
        _, group_idx = torch.unique(id, return_inverse=True)

        # scatter_softmax perform softmax independly per each unique group. 
        # Group assigments are passed as second argument. 
        weights = torch_scatter.scatter_softmax(self.linear1(x), group_idx, dim=1)
        
        y = torch_scatter.scatter_sum(self.linear2(x) * weights, group_idx, dim=1)

        if self.expand:
            return self.self.linear3(y)[:,jx]
            
        return self.linear3(y)

def neighbours(patch_idx, target_frame, device, range = 1):
    
    base = torch.stack([patch_idx, target_frame], dim=1) # shape (n, 2)
    prev = torch.stack([patch_idx, target_frame - range], dim=1)
    next = torch.stack([patch_idx, target_frame + range], dim=1)

    base = base.unsqueeze(0).permute(0, 2, 1) # shape  (1, 2, n)
    prev = prev.unsqueeze(-1) # shape (n, 2, 1)
    next = next.unsqueeze(-1) # shape (n, 2, 1)

    # search in past frames
    mask = (prev == base) # shape (n, 2, n)
    # if mask[i, :, k].all() == True, that means position i in prev (prev[i, :]) exist in base on position k (base[k, :]
    match = mask.all(dim=1)
    i_prev, k_prev = match.nonzero(as_tuple=True) # each shape (t,) where t is number of matches and prev[i_prev, :] == base[k_prev, :]

    # search in future frames
    mask = (next == base) # shape (n, 2, n)
    # if mask[i, :, k].all() == True, that means position i in prev (prev[i, :]) exist in base on position k (base[k, :]
    match = mask.all(dim=1)
    i_next, k_next = match.nonzero(as_tuple=True) # each shape (t,) where t is number of matches and prev[i_next, :] == base[k_next, :]

    # create tensor where if there is no match, set -1.0
    prev_idx = torch.full(patch_idx.shape, -1, device=device, dtype=torch.long)
    next_idx = torch.full(patch_idx.shape, -1, device=device, dtype=torch.long)

    prev_idx[i_prev] = k_prev
    next_idx[i_next] = k_next
    
    return prev_idx, next_idx
    
