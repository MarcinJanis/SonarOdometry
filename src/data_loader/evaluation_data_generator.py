
import pandas as pd
import numpy as np 

import os
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io



class DataGenerator():

    def __init__(self, data_dir_pth, device, transforms = None):

        self.dir_pth = data_dir_pth

        self.csv_path = os.path.join(self.dir_pth, 'sequence.csv')
        self.fls_path = os.path.join(self.dir_pth, 'fls')

        self.time = pd.read_csv(self.csv_path, usecols=['timestamp'])
        self.pose = pd.read_csv(self.csv_path, usecols=['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w'])

        self.transforms = transforms

        self.device = device

    def get_sample(self, idx, return_visu=False):

        # get sonar frame
        frame_pth = os.path.join(self.fls_path, f'{idx}.png')
        frame = io.read_image(frame_pth).float()
        frame = frame / 255.0
        frame = frame.unsqueeze(0).unsqueeze(0).to(self.device)

        # get other data
        t = torch.tensor(self.time.iloc[idx].values, dtype = torch.float, device = self.device)
        pose = torch.tensor(self.pose.iloc[idx].values, dtype = torch.float, device = self.device)
        pose[3:7] = F.normalize(pose[3:7], p=2, dim=-1)

        if return_visu:
            frame_np = frame.cpu().numpy() * 255
            return t, frame, frame_np, pose
        else:
            return t, frame, pose
            

    def get_len(self):
        return len(self.time)
        
