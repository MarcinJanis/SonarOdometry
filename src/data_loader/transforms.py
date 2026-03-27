import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Gamma
from torchvision.transforms import v2

# === PARAMETERS === 

# Parameters determined by analysis of Aracati2017 data set. 
# Analysis: ./notebooks/test/sonar_noise.ipynb
# Aracati2017 dataset: see in README

SPECKLE_NOISE_CONCENTRATION = 7.250794635405014
SPECKLE_NOISE_RATE = 7.250794635405014
SPECKLE_NOISE_UPSAMPLE_FACTOR = 4 # 2 for smaller noise


# ==================
class SpeckleNoise(nn.Module):
    def __init__(self, concentration=1.0, rate=1.0):
        super().__init__()

        self.concentration = concentration
        self.rate = rate
        # self.blur = blur
        # self.upsample_factor = upsample_factor
        
    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.view(b*n, c, h, w)

        concentration = torch.tensor([self.concentration], device=x.device, dtype=x.dtype)
        rate = torch.tensor([self.rate], device=x.device, dtype=x.dtype)
        
        gamma_dist = Gamma(concentration, rate)

        if torch.rand(1).item() > 0.5:
            upsample_factor = 2
        else:
            upsample_factor = 4

        speckle_noise = gamma_dist.sample((b*n, c, h//upsample_factor,  w//upsample_factor)).squeeze(-1)
        
        speckle_noise_upsample = F.interpolate(
            speckle_noise,
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        )

        # avg_kernel = torch.ones((b*n, c, 3, 3), dtype=torch.double) * self.blur
        # speckle_noise_filtered = F.conv2d(speckle_noise, avg_kernel, padding=1)

        x = torch.clamp(x * speckle_noise_upsample, min=0.0, max=1.0)
        x = x.view(b, n, c, h, w)
        return x
    import torch
import torch.nn as nn
import math

class RayArtifacts(nn.Module):
    def __init__(self, a_min, a_max, w_min, w_max, num_rays=5, probability=1.0):
        super().__init__()
        self.a_min = a_min 
        self.a_max = a_max
        self.w_min = w_min
        self.w_max = w_max
        self.num_rays = num_rays
        self.probability = probability

    def forward(self, x):
   
        if torch.rand(1).item() > self.probability:
            return x

        b, n, c, h, w = x.shape
        device = x.device
        dtype = x.dtype

        xaxis = torch.linspace(0, 2 * math.pi, w, device=device, dtype=dtype)
        
        amplitude = (torch.rand(self.num_rays, device=device, dtype=dtype) * (self.a_max - self.a_min) + self.a_min)
        omega = torch.randint(low=self.w_min, high=self.w_max + 1, size=(self.num_rays,), device=device)

        rays_matrix = amplitude.unsqueeze(1) * torch.sin(omega.unsqueeze(1).to(dtype) * xaxis)
        
        rays_1d = rays_matrix.sum(dim=0)
        rays_exp = rays_1d.view(1, 1, 1, 1, w).expand(b, n, c, h, w)

        return torch.clamp(x + rays_exp, 0.0, 1.0)


SonarDatasetTranforms = v2.Compose([
    RayArtifacts(0.0, 0.03, 0, 10, num_rays=4, probability=0.85),
    v2.GaussianBlur(kernel_size=(7, 5), sigma=(1.0, 2.0)),
    SpeckleNoise(concentration=SPECKLE_NOISE_CONCENTRATION, rate=SPECKLE_NOISE_RATE)])


