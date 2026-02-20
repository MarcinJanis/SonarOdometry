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

        self.corr_net = nn.Sequential(
            nn.Linear(corr_input_dim, hidden_state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.LayerNorm(hidden_state_dim, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state_dim, hidden_state_dim)
        )

        self.norm = nn.LayerNorm(hidden_state_dim, eps=1e-3)

        self.c1 = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state_dim, hidden_state_dim))

        self.c2 = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_state_dim, hidden_state_dim))

        self.patches_agg = SoftAgg(dim=hidden_state_dim, expand=True)
        self.edges_agg = SoftAgg(dim=hidden_state_dim, expand=True)

        # recurrent net here:
        self.gru = nn.Sequential(
            nn.LayerNorm(hidden_state_dim, eps=1e-3),
            GatedResidual(hidden_state_dim), 
            nn.LayerNorm(hidden_state_dim, eps=1e-3),
            GatedResidual(hidden_state_dim)
        )
        

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(hidden_state_dim, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(hidden_state_dim, 2),
            GradientClip(),
            nn.Sigmoid())


    
    def forward(self, h, flow, corr, ctx, source_frame_idx, target_frames_idx, patches_idx, device):
            
        # --- update hidden state with new data --- 
        corr = self.corr_net(corr) # process correlation tensor
        h = h + ctx + corr # add to hidden state
        h = self.norm(h) # normalize

        # for each edge find edge idx, where same patch is matched with previous or next target frame in time. 
        prev_idx, next_idx = neighbours(patches_idx, target_frames_idx, device = device, range = 1)

        prev_mask = (prev_idx >= 1).float #.reshape(1, -1, 1)
        next_mask = (next_idx >= 1).float #.reshape(1, -1, 1)

        h = h + self.c1(prev_mask * h[prev_idx, :]) # add to hidden state information about temporal patches neighbours 
        h = h + self.c2(next_mask * h[next_idx, :]) # add to hidden state information about temporal patches neighbours 

        h = h + self.patches_agg(h, patches_idx)
        # h = h + self.edges_agg(h, source_frame_idx*12345 + target_frames_idx)
        h = h + self.edges_agg(h, source_frame_idx*(target_frames_idx.max() + 1) + target_frames_idx)

        h = self.gru(h)

        delta = self.d(h) # projection correction (dx, dy)
        weights = self.w(h) # correction weights, confidence 
        
        return h, delta, weights

# =========


class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)


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
        self.linear3 = nn.Linear(self.dim, self.dim)

    def forward(self, x, id):

        # assign each element of id tensor to group, a unique group assigment is achived
        # unique, idx = torch.unique(input, return_inverse=True) # returns: unique - unique values that occurs in input tensor, idx - for each element idx of value from unique is assigned 
        _, group_idx = torch.unique(id, return_inverse=True)

        # scatter_softmax perform softmax independly per each unique group. 
        # Group assigments are passed as second argument. 
        weights = torch_scatter.scatter_softmax(self.linear1(x), group_idx, dim=1)
        
        y = torch_scatter.scatter_sum(self.linear2(x) * weights, group_idx, dim=1)

        if self.expand:
            return self.self.linear3(y)[:,group_idx]
            
        return self.linear3(y)

def neighbours_broadcast(patch_idx, target_frame, device, range = 1): # alternative
    
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

def neighbours(patch_idx, target_frame, device, range = 1):
    
    sort_key = patch_idx * (target_frame.max() + 1) + target_frame # copress patch idx and frame idx into one value to be sorted
    sorted_keys, indices = torch.sort(sort_key) # sort 

    # print(sorted_keys)
    rev_indices= torch.argsort(indices) # to restore orginal order 

    # print(sorted_keys[range:])
    # print((sorted_keys - range)[:-range])

    has_prev = sorted_keys[range:] == (sorted_keys + range)[:-range] 
    has_next = sorted_keys[:-range] == (sorted_keys - range)[range:] 
  
    # print(f'has_next: {has_next}')
    # print(f'has_prev: {has_prev}')

    prev_indices = torch.full(sorted_keys.shape, -1, device=device, dtype=torch.long)
    next_indices = torch.full(sorted_keys.shape, -1, device=device, dtype=torch.long)

    # print(f'prev indices: {prev_indices}')
    # print(f'next indices: {next_indices}')

    # print()
    prev_indices[range:][has_prev] = indices[torch.nonzero(has_prev).squeeze(-1).long()]
    next_indices[:-range][has_next] = indices[torch.nonzero(has_next).squeeze(-1).long() + 1]

    # print(f'prev indices: {prev_indices}')
    # print(f'next indices: {next_indices}')
    prev_indices = prev_indices[rev_indices]
    next_indices = next_indices[rev_indices]

    return prev_indices, next_indices 

