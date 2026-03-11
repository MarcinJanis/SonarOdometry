
import pandas as pd
import numpy as np 

import os
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io

class SonarSimDataset(Dataset):
    def __init__(self, root_dir, window_size, transform=None):

        self.root_dir = root_dir 
        self.window_size = window_size
        self.transform = transform

        self.time = {} # time vector
        self.traj_gt = {} # trajectory 
        self.depth = {}
        self.seq_strat_idx = [] # idx of starting frame 
        self.seq_len = {} # sample number for each sequence

        for dir in os.scandir(self.root_dir):
            
            seq_name = dir.name
            seq_path = dir.path

            if seq_name == 'aracati': continue

            csv_path = os.path.join(seq_path, 'sequence.csv')
            fls_path = os.path.join(seq_path, 'fls')

            time = pd.read_csv(csv_path, usecols=['timestamp'])
            pose = pd.read_csv(csv_path, usecols=['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w'])
            depth = pd.read_csv(csv_path, usecols=['dvl_alt'])

            self.time[seq_name] = time
            self.traj_gt[seq_name] = pose 
            self.depth[seq_name] = depth
            seq_len = len(time)
            self.seq_len[seq_name] = seq_len

            if seq_len > self.window_size:
                for idx in range(seq_len - self.window_size + 1): # imo bez + 1
                    self.seq_strat_idx.append((seq_name, idx)) 
   
            

    def __len__(self):
        return len(self.seq_strat_idx)

    def __getitem__(self, idx):
        
        seq_name, start_idx = self.seq_strat_idx[idx]
        end_idx = start_idx + self.window_size

        
        time = self.time[seq_name].iloc[start_idx:end_idx]#[start_idx:end_idx]
        trajectory = self.traj_gt[seq_name].iloc[start_idx:end_idx, :]
        depth = self.depth[seq_name].iloc[start_idx:end_idx]

        time = torch.from_numpy(time.values).float()
        trajectory = torch.from_numpy(trajectory.values).float()
        depth = torch.from_numpy(depth.values).float()

        imgs = []
        for i in range(self.window_size):
            img_pth = os.path.join(self.root_dir, seq_name, 'fls', f'{start_idx + i}.png')
            # img_np = cv2.imread(img_pth, 0)
            # tensor = torch.tensor(img_np)
            tensor = io.read_image(img_pth)
            imgs.append(tensor)

        series = torch.stack(imgs).float()

        # norm 
        series = series / 255.0

        return series, time, trajectory, depth

    def print_info(self):
        print('='*40)
        print('Dataset content:')
        print(f'Number of sequences: {len(self.seq_len)}')
        print(f'Total frames number: {sum(self.seq_len.values())}')
        for key, val in self.seq_len.items():
            print(f'Seq: {key}: {val} frames.')
        print('='*40)
        


def visu_format(x):
    x.detach().clone()
    n, c, h, w = x.shape
    output = []
    for frame_num in range(n):
        frame = x[frame_num] # (c, h, w)
        frame = frame.permute(1, 2, 0)
        frame_np = frame.cpu().numpy() * 255
        output.append(frame_np)
    return output 



# data_headers = {
#     'idx':0,        # idx od sample
#     't':1,          # time stamp
#     'pos_x':2,      # position: x-axis
#     'pos_y':3,      # position: y-axis
#     'pos_z':4,      # position: z-axis
#     'orient_x':5,   # orientation: x-axis, roll
#     'orient_y':6,   # orientation: y-axis, pitch
#     'orient_z':7,   # orientation: 
#     'orient_w':8,   # orientation: 
#     'dvl_vel_x':9,  # linear velocity: x-axis, DVL 
#     'dvl_vel_y':10, # linear velocity: y-axis, DVL 
#     'dvl_vel_z':11, # linear velocity: ?-axis, DVL 
#     'dvl_vel_w':12, # linear velocity: ?-axis, DVL 
#     'alt':13,       # altitude, height over ground DVL
#     'imu_orient_x':14,   # orientation: x-axis, roll, IMU
#     'imu_orient_y':15,   # orientation: y-axis, pitch, IMU
#     'imu_orient_z':16,   # orientation: z-axis, yaw, IMU
#     'imu_gyro_x':17,     # angular velocity: x-axis, roll, IMU
#     'imu_gyro_y':18,     # angular velocity: y-axis, pitch, IMU
#     'imu_gyro_z':19,     # angular velocity: x-axis, yaw, IMU
#     'imu_accel_x':20,    # linear acceleration: x-axis, IMU
#     'imu_accel_y':21,    # linear acceleration: x-axis, IMU
#     'imu_accel_z':22,    # linear acceleration: x-axis, IMU
# }