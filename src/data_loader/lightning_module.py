
import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from .metrics import pose_err
from .metrics import eval_metrics

# supervised_traning_param = {
#     'freeze_poses_steps':4,
#     'init_pose_max_noise':2.0,
#     'loss_weight_trans':1.0,
#     'loss_weight_rot':1.0,
#     'loss_weight_proj_r':1.0,
#     'loss_weight_proj_theta':1.0
# }

# selfsupervised_traning_param = {
#     'loss_weight_proj_r':1.0,
#     'loss_weight_proj_theta':1.0
# }
class DPSO_LightningModule(pl.LightningModule):
    def __init__(self, model, mode, traning_param):
        super().__init__()
        self.model = model

        self.save_hyperparameters()

        self.mode = mode
        self.traning_param = traning_param

        if mode == 'supervised':
            self.supervised = True
            self.freeze_poses_steps = traning_param['freeze_poses_steps']
            self.loss_w_trans = traning_param['loss_weight_trans']
            self.loss_w_rot = traning_param['loss_weight_rot']
        else:
            self.supervised = False

        
        self.init_poses_noise = traning_param['init_pose_max_noise']

        self.loss_w_proj_r = traning_param['loss_weight_proj_r']
        self.loss_w_proj_theta = traning_param['loss_weight_proj_theta']
        self.loss_w_weights = traning_param['loss_weight_weights'] 

        self.freeze_delta_loss_step = traning_param['freeze_delta_loss_step']
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # Scheduluer, który reaguje na to, co faktycznie dzieje się z siecią
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',      
            factor=0.5,      
            patience=3,      
            min_lr=1e-6      
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # check name
                # "interval": "step",   # check each 1000 steps
                # "frequency": 500 # same as in trainer val_check_interval
            },
        }


    def training_step(self, batch, batch_idx):

        if self.supervised: 
            # freeze poses 
            if self.global_step < self.freeze_poses_steps:
                freeze_poses = True
            else:
                freeze_poses = False
        else:
            freeze_poses = False

        loss_trans = 0.0
        loss_rot = 0.0
        loss_theta = 0.0
        loss_r = 0.0

        fls_series, time, trajectory_gt, depth_gt = batch
        
        pred = self.model(frames=fls_series, 
                        timestamp=time, 
                        poses_gt=trajectory_gt, 
                        depth_gt=depth_gt, 
                        supervised=self.supervised, 
                        freeze_poses=freeze_poses, 
                        init_poses_noise=self.init_poses_noise, 
                        debug_logger=False)

        for k, (pred_poses, target_projection, predicted_projection, valid_mask, weights) in enumerate(pred):
            
            if self.supervised: # is supervised, compute ATE, as reference metric but not add to to loss fcn
                # pose eror - mean from absolute pose and rotation error
                trans_err, rot_err = pose_err(pred_poses, trajectory_gt)
                # accumulate loss components
                loss_trans += trans_err 
                loss_rot += rot_err
            
            # --- projection error --- 
            # supervised - between prediction and gt
            # selfsupervised - between prediction and optimized value by BA
            valid_edges_num = torch.sum(valid_mask) + 1e-6

            # === Smooth L1 Loss with valid mask ===
            err_raw = F.smooth_l1_loss(predicted_projection, target_projection, reduction='none', beta=1.0)
            # beta - err value when L2 is used instead of L1 for specific edge
            # L1 has constant gradient (1 or -1), so outliers and huge error error don't destroy loss fcn
            # L2 has gradient proportional to it's value so allows to find minimum for small loss value
            
            # === Mean absolute error with valid mask ===
            # err_raw = torch.abs(target_projection - predicted_projection)
            
            # ==== Weights loss ===
           
            if self.global_step < self.freeze_delta_loss_step:
                weight_target_loss = F.mse_loss(weights, torch.ones_like(weights))
                loss_weighted = err_raw + 0.1 * weight_target_loss.unsqueeze(-1)
            else:
                loss_weighted = weights * err_raw - self.loss_w_weights * torch.log(weights + 1e-6)

            # === Connenct err with weights and valid mask ===
            patch_proj_err = valid_mask.unsqueeze(-1) * loss_weighted

            proj_x_err = torch.sum(patch_proj_err[:, 0]) / valid_edges_num # theta err 
            proj_y_err = torch.sum(patch_proj_err[:, 1]) / valid_edges_num # r err

            # accumulate loss components
            loss_theta += proj_x_err 
            loss_r += proj_y_err 

        # compute loss 
        k_total = k + 1

        loss_theta = loss_theta / k_total
        loss_r = loss_r / k_total

        if self.supervised: 
            loss_trans = loss_trans / k_total
            loss_rot = loss_rot / k_total

            self.log_dict({'loss_translation':loss_trans, 'loss_rotation':loss_rot, 'loss_projection_theta':loss_theta, 'loss_projection_r':loss_r, 'loss_weighted':loss_weighted}, on_step=True, on_epoch=False, logger=True)

            total_loss = self.loss_w_proj_r * loss_r + \
                         self.loss_w_proj_theta * loss_theta
            
        else:

            self.log_dict({'loss_projection_theta':loss_theta, 'loss_projection_r':loss_r}, on_step=True, on_epoch=False, logger=True)

            total_loss = self.loss_w_proj_r * loss_r + \
                         self.loss_w_proj_theta * loss_theta

        self.log('total_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return total_loss
        

    def validation_step(self, batch, batch_idx):
        
        # freeze poses 
        freeze_poses = False

        fls_series, time, trajectory_gt, depth_gt = batch

        pred = self.model(frames=fls_series, 
                        timestamp=time, 
                        poses_gt=trajectory_gt, 
                        depth_gt=depth_gt, 
                        supervised=self.supervised, 
                        freeze_poses=freeze_poses, 
                        init_poses_noise=0.0, 
                        debug_logger=False)

        pred_poses, target_projection, predicted_projection, valid_mask, weights = pred[-1]
        
        valid_edges_num = torch.sum(valid_mask) + 1e-6

        # === Smooth L1 Loss with valid mask ===
        err_raw = F.smooth_l1_loss(predicted_projection, target_projection, reduction='none', beta=1.0)
        # beta - err value when L2 is used instead of L1 for specific edge
        # L1 has constant gradient (1 or -1), so outliers and huge error error don't destroy loss fcn
        # L2 has gradient proportional to it's value so allows to find minimum for small loss value
        
        # === Mean absolute error with valid mask ===
        # err_raw = torch.abs(target_projection - predicted_projection)
        
        # ==== Weights loss === 
        loss_weighted = weights * err_raw - 0.2 * torch.log(weights + 1e-6)

        # === Connenct err with weights and valid mask ===
        patch_proj_err = valid_mask.unsqueeze(-1) * err_raw * loss_weighted

        proj_x_err = torch.sum(patch_proj_err[:, 0], dim=-1) / valid_edges_num
        proj_y_err = torch.sum(patch_proj_err[:, 1], dim=-1) / valid_edges_num

        metrics = eval_metrics(pred_poses.detach().cpu().numpy(), 
                               trajectory_gt.detach().cpu().numpy(),
                               align=False, align_init_pt_only=True, add_data_series=False)
        
        metrics['projection_err_theta_val'] = proj_x_err
        metrics['projection_err_r_val'] = proj_y_err
        metrics['loss_weighted'] = loss_weighted

        total_loss = self.loss_w_proj_r * proj_y_err + \
                         self.loss_w_proj_theta * proj_x_err
        
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)

        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        
        
