import torch
import torch.nn as nn
import torch.nn.functional as F





class BoundleAdjustment(nn.Module):
    def __init__(self):
        super().__init__()


        # --- to be optimized ---
        self.poses = None
        self.elevation_ang = None

        # --- optimization based on ---
        self.delta = None
        self.weights = None 


    def init_ba(self):
        
        

    def forward(self):
        pass
