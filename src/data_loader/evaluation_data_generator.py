
import pandas as pd
import numpy as np 

import os
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fast')

class DataGenerator():

    def __init__(self, data_dir_pth, device, transforms = None):

        self.dir_pth = data_dir_pth

        self.csv_path = os.path.join(self.dir_pth, 'sequence.csv')
        self.fls_path = os.path.join(self.dir_pth, 'fls')

        self.time = pd.read_csv(self.csv_path, usecols=['timestamp'])
        self.pose = pd.read_csv(self.csv_path, usecols=['pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w'])

        self.transforms = transforms

        self.device = device

        # --- Data ---
        self.predict_traj = {}
        self.pts3d = None

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
            return t, frame, pose, frame_np
        else:
            return t, frame, pose
            

    def get_len(self):
        return len(self.time)
        

    def read_estim_trajectory(self, csv_pth, label):
        self.predict_traj[label] = pd.read_csv(csv_pth, usecols=['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']) # pose_no,t,x,y,z,qx,qy,qz,qw

    def read_pts(self, csv_pth):
        self.pts3d = pd.read_csv(csv_pth, usecols=['x', 'y', 'z'])

    def generate_trajectory_map_2d(self, plane = 'xy', show = {'gt':True,'traj':True,'pts':True}, start = 0, end = 1, colors = ('red', 'green', 'blue', 'orange'), traj_width = 4, pt_size = 3, save_to_file = None):
        # show_traj = (gt, primary, secondary)
        n_max = self.get_len()

        start_idx = int(start*n_max)
        end_idx = int(end*n_max)

        if plane == 'xy':
            ax1 = 0
            ax2 = 1
            ax3 = 2
        elif plane == 'yz':
            ax1 = 1
            ax2 = 2
            ax3 = 0
        elif plane == 'xz':
            ax1 = 0
            ax2 = 2
            ax3 = 1

        # create plot 
        fig, ax = plt.subplots(figsize=(20, 20))

        # --- Ground truth ---
        if show['gt']:
            
            gt_x = self.pose.iloc[start_idx:end_idx].values[:, ax1]
            gt_y = self.pose.iloc[start_idx:end_idx].values[:, ax2]
            gt_lbl = 'gt'

            ax.plot(gt_x, gt_y, color='black', linewidth=traj_width, alpha=0.6, label=gt_lbl)
            ax.scatter(gt_x[0], gt_y[0], color='black', zorder=5)
            ax.scatter(gt_x[-1], gt_y[-1], color='black', zorder=5)
            
        # --- trajectories --- 
        predict_traj = list(self.predict_traj.values())
        predict_traj_lbl =  list(self.predict_traj.keys())

        if show['traj']:
            for k in range(len(predict_traj)):
                traj1_x = predict_traj[k].iloc[start_idx:end_idx].values[:, ax1]
                traj1_y = predict_traj[k].iloc[start_idx:end_idx].values[:, ax2]
                traj1_lbl = predict_traj_lbl[k]

                ax.plot(traj1_x, traj1_y, color=colors[k], linewidth=traj_width, alpha=1, label=traj1_lbl)
                ax.scatter(traj1_x[0], traj1_y[0], color='black', zorder=5)
                ax.scatter(traj1_x[-1], traj1_y[-1], color='black', zorder=5)

        if show['pts']:
            x = self.pts3d.iloc[start_idx:end_idx].values[:, ax1]
            y = self.pts3d.iloc[start_idx:end_idx].values[:, ax2]
            z = self.pts3d.iloc[start_idx:end_idx].values[:, ax3]
            ax.scatter(x, y, c=z, cmap='gray_r', s=pt_size)
            pass

        
        # general plot setup 
        ax.set_title(f"Trajectory estimation.")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.7)
        ax.grid(which='minor', linestyle=':', linewidth='0.3', color='black', alpha=0.5)
        ax.legend()

        plt.show()

        # save to file
        if not save_to_file is None:
            plt.savefig(
                save_to_file, 
                dpi=300,                
                bbox_inches='tight',    
                pad_inches=0.1,        
                transparent=False       
            )

    def generate_trajectory_map_3d(self, show = {'gt':True,'traj':True,'pts':True}, start = 0, end = 1, colors = ('red', 'green', 'blue', 'orange'), traj_width = 4, pt_size = 3, save_to_file = None):
        
        n_max = self.get_len()

        start_idx = int(start * n_max)
        end_idx = int(end * n_max)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        # --- Ground Truth ---
        if show['gt']:
            gt_x = self.pose.iloc[start_idx:end_idx]['pos_x'].values
            gt_y = self.pose.iloc[start_idx:end_idx]['pos_y'].values
            gt_z = self.pose.iloc[start_idx:end_idx]['pos_z'].values
            
            ax.plot(gt_x, gt_y, gt_z, color='black', label='Ground Truth', linewidth=traj_width,  alpha=0.7)
            
            ax.scatter(gt_x[0], gt_y[0], gt_z[0], color='black', s=50, label='Start')
            ax.scatter(gt_x[-1], gt_y[-1], gt_z[-1], color='black', s=50, label='End')

        # --- Trajectory ---
        predict_traj = list(self.predict_traj.values())
        predict_traj_lbl =  list(self.predict_traj.keys())

        if show['traj']:
            for k in range(len(predict_traj)):
                traj_x = predict_traj[k].iloc[start_idx:end_idx].values[:, 0]
                traj_y = predict_traj[k].iloc[start_idx:end_idx].values[:, 1]
                traj_z = predict_traj[k].iloc[start_idx:end_idx].values[:, 2]
 

                ax.plot(traj_x, traj_y, traj_z, color=colors[k], label=predict_traj_lbl[k], linewidth=traj_width,  alpha=0.7)
                ax.scatter(traj_x[0], traj_y[0], traj_z[0], color=colors[k], s=50, label='Start')
                ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], color=colors[k], s=50, label='End')

        # --- 3D Points ---
        if show['pts'] and self.pts3d is not None:
            px = self.pts3d.iloc[start_idx:end_idx]['x'].values
            py = self.pts3d.iloc[start_idx:end_idx]['y'].values
            pz = self.pts3d.iloc[start_idx:end_idx]['z'].values
            
            # Mapping color to depth (Z) exactly like we discussed for 2D
            ax.scatter(px, py, pz, c=pz, cmap='viridis', s=pt_size, alpha=0.5)

        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title("3D Trajectory and Point Cloud")
        ax.legend()
        
        plt.show()

    def generate_time_series():
        pass
