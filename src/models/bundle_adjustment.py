import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import transorm_points_coords, transform_matrix, projection_type

import pypose as pp


class BoundleAdjustment(nn.Module):
    def __init__(self, poses, patch_coords):
        super().__init__()

        poses_se3 = pp.SE3(poses)
        self.poses = pp.Patameter(poses_se3)
        
        self.patch_coords = patch_coords[patch_coords[:, 2]
        self.elevation_angle = pp.Patameter(patch_coords[:, -1])


    def init_ba(self, poses_idx, patch_idx, delta, weights):
        
        self.poses_idx = poses_idx
        self.patch_idx = patch_idx

        # --- get poses and patch coords ---
        poses = self.poses[self.poses_idx]
        patch_coords = self.patch_coords[self.patch_idx]

        self.weights = weights
        
        with torch.no_grad():
            # --- transform points --- 
            T = transform_matrix(self.edge_poses)
            target_coords  = T @ target_coords # złożyć target coords z tych dóch wcześniejsyzch 
            target_coords = transorm_points_coords(target_coords, projection_type.CARTESIAN2POLAR)
            
            # --- add corrections ---
            target_coords = target_coords + delta 
            self.target_coords = target_coords.detach()
            
        # self.edge_poses = edge_poses# self.poses[edge_poses_idx]
        # # edge_elevation_ang = self.patch_state[edge_patch_idx][:, -1]
        # self.edge_patch_coords = edge_patch_coords #self.patch_state[edge_patch_idx]
        # self.elevation_angle = edge_patch_coords[:, -1]
        
        
        # --- transform points to sonar coord system 
        

        # # --- calc projection error --- 
        # residual = (projected_coords - target_coords) * weights  
        
        # return residual
        

    def forward(self):

        # --- get poses and coords ---
        poses = self.poses[self.poses_idx]
        proj_coords = self.patch_coords[self.patch_idx]
        
        T = transform_matrix(poses)
        proj_coords  = T @ proj_coords
        proj_coords = transorm_points_coords(proj_coords, projection_type.CARTESIAN2POLAR)

        # calc projection error
        resiudal = (proj_coords - self.target_coords) * self.weights 
    
        return reisudal 

    def run(self):
        pass
###

'''
how to optimize:



import pypose as pp
import torch

# 1. Inicjalizacja Twojego modelu (stanu początkowego)
# Zwróć uwagę, że model sam w sobie przechowuje poses i inv_depths jako pp.Parameter
ba_model = DPVO_BundleAdjustment(init_poses, init_inv_depths, intrinsics)

# 2. Definicja optymalizatora
# Przekazujesz cały model do optymalizatora LM. PyPose sam znajdzie pp.Parameter.
optimizer = pp.optim.LM(ba_model)

# 3. Parametry pętli
max_iterations = 10  # Zwykle w BA wystarcza od kilku do kilkunastu iteracji
prev_loss = float('inf')

print("Rozpoczynam Bundle Adjustment...")

# 4. Pętla optymalizacyjna
for i in range(max_iterations):
    # Wywołanie optimizer.step() automatycznie:
    # - przepuszcza dane przez ba_model.forward(...)
    # - liczy Jacobiany dla poses i inv_depths
    # - aktualizuje poses i inv_depths
    # WAŻNE: argumenty w step() muszą dokładnie odpowiadać argumentom Twojego forward()!
    loss = optimizer.step(target_coords, weights, cam_idx, patch_idx)
    
    print(f"Iteracja {i}, Koszt: {loss.item():.4f}")
    
    # Warunek stopu (wczesne wyjście, jeśli błąd przestał znacząco spadać)
    if abs(prev_loss - loss.item()) < 1e-5:
        print("Zbieżność osiągnięta!")
        break
    prev_loss = loss.item()

# 5. Wyciągnięcie zoptymalizowanych wyników!
# Parametry w ba_model zaktualizowały się w miejscu (in-place). 
# Wystarczy je wyciągnąć i odciąć gradienty (.detach())
optimized_poses = ba_model.poses.detach()
optimized_inv_depths = ba_model.inv_depths.detach()

print("Gotowe! Nowe pozy obliczone.")
'''
