import torch
import torch.nn as nn
import torch.nn.functional as F





class Update(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        # --- get model configuration ---
        self.fmap_c = self.model_cfg.FEATURES_OUTPUT_CH
        self.corr_neighbour = self.model_cfg.CORR_NEIGHBOUR
        self.patch_size = self.model_cfg.PATCH_SIZE 
        
        # correlation preprocess net
        corr_input_dim = self.fmap_c*self.corr_neighbour*self.corr_neighbour*self.patch_size*self.patch_size
        corr_output_dim = self.model_cfg. #TODO
        self.corr_net = nn.Sequential(
            nn.Linear(corr_input_dim, corr_output_dim)
        )

   
        

    
    def forward():
        pass


