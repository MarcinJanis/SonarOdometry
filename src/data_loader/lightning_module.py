import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from .metrics import eval_metrics

# traning_param = {
# 'freeze_poses_steps':0.0,
# 'init_pose_max_noise':0.0,
# 'freeze_weights_global_steps':0.0,
# 'lr_scheduler_patience':0.0,
# 'lr_scheduler_interval_global_steps':0.0,
# 'weights_loss_gamma':0.7
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
            self.freeze_poses = traning_param['freeze_poses']
        else:
            self.supervised = False

        self.init_poses_noise_trans = traning_param['init_pose_max_noise_trans']
        self.init_poses_noise_rot = traning_param['init_pose_max_noise_rot']
        self.gamma = traning_param['weights_loss_gamma']

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',      
            factor=0.5,      
            patience=self.traning_param['lr_scheduler_patience'], 
            min_lr=1e-6      
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # check name
                "interval": "step",   
                "frequency": self.traning_param['lr_scheduler_interval_global_steps'] # global steps
            },
        }
    
    def training_step(self, batch, batch_idx):

        # freeze poses 
        # if self.supervised: 
        #     if self.freeze_poses:
        #         freeze_poses = True
        #     else:
        #         freeze_poses = False
        # else:
        #     freeze_poses = False

        # loss_trans = 0.0
        # loss_rot = 0.0
        # loss_theta = 0.0
        # loss_r = 0.0

        # --- pass data through net --- 
        fls_series, time, trajectory_gt, depth_gt = batch
        
        pred = self.model(frames=fls_series, 
                        timestamp=time, 
                        poses_gt=trajectory_gt, 
                        depth_gt=depth_gt, 
                        supervised=self.supervised, 
                        freeze_poses=self.freeze_poses,
                        init_poses_noise=(self.init_poses_noise_trans, self.init_poses_noise_rot),
                        debug_logger=False)


        # --- iterate over each prediction --- 
        max_pred_iter = len(pred)

        total_loss = 0.0

        for k, (pred_poses, target_projection, predicted_projection, valid_mask, weights, delta) in enumerate(pred):
            
            # --- reprojection error --- 
            # supervised - between prediction and gt
            # selfsupervised - between prediction and optimized value by BA
            valid_edges_num = torch.sum(valid_mask) + 1e-6

            # --- Smooth L1 Loss with valid mask ---
            # use L1 norm, when err > beta, use L2 if err < beta
            err_raw = F.smooth_l1_loss(predicted_projection, target_projection, reduction='none', beta=2.5)
          
            # --- weights - Kandell Loss --- 
            if self.freeze_poses: 
                # do not use weighted error
                loss_weighted = err_raw
            else:
                loss_weighted = torch.exp(-weights) * err_raw + weights

           
            #  --- mask weighted error - keep gradient for valid edges only --- 
            patch_proj_err = valid_mask.unsqueeze(-1) * loss_weighted

            proj_x_err = torch.sum(patch_proj_err[:, 0]) / valid_edges_num # r err 
            proj_y_err = torch.sum(patch_proj_err[:, 1]) / valid_edges_num # theta err

            # accumulate loss components

            step_loss = proj_x_err + proj_y_err
            weight_step = self.gamma ** (max_pred_iter - k - 1)
            total_loss += weight_step * step_loss
    
        # --- compute total loss from all predictions --- 
        k_total = k + 1
        total_loss = total_loss / k_total

        # --- log stats ---

        self.log_dict({'total_loss':total_loss, 'mean_projection_err_r':proj_x_err, 'mean_projection_err_theta':proj_y_err, 
                       'mean_weights_r':torch.mean(weights[:, 0]), 'mean_weights_theta':torch.mean(weights[:, 1])}, 
                       on_step=True, on_epoch=False, logger=True)

        return total_loss

       
    def validation_step(self, batch, batch_idx):
        
        freeze_poses = False 

        fls_series, time, trajectory_gt, depth_gt = batch

        pred = self.model(frames=fls_series, 
                        timestamp=time, 
                        poses_gt=trajectory_gt, 
                        depth_gt=depth_gt, 
                        supervised=self.supervised, 
                        freeze_poses=freeze_poses, 
                        init_poses_noise=(self.init_poses_noise_trans, self.init_poses_noise_rot), 
                        debug_logger=False)

        pred_poses, target_projection, predicted_projection, valid_mask, weights_s, delta = pred[-1]
        
        valid_edges_num = torch.sum(valid_mask) + 1e-6

        # --- Smooth L1 Loss ---
        err_raw = F.smooth_l1_loss(predicted_projection, target_projection, reduction='none', beta=1.0)
        
        # do not use weighted loss
        
        # --- valid mask and total loss ---
        patch_proj_err = valid_mask.unsqueeze(-1) * err_raw

        proj_x_err = torch.sum(patch_proj_err[:, 0], dim=-1) / valid_edges_num
        proj_y_err = torch.sum(patch_proj_err[:, 1], dim=-1) / valid_edges_num

        val_loss = proj_y_err + proj_x_err

        # --- log metric --- 
        trajectory_gt_sonar = self.model.calib.pose(trajectory_gt)
        metrics = eval_metrics(pred_poses, trajectory_gt_sonar, reduction = 'mean') 
        metrics['mean_abs_weights_r'] = torch.mean(torch.abs(weights_s[:, 0]))
        metrics['mean_abs_weights_theta'] = torch.mean(torch.abs(weights_s[:, 1]))
        metrics['mean_abs_delta_r'] = torch.mean(torch.abs(delta[:, 0]))
        metrics['mean_abs_delta_theta'] = torch.mean(torch.abs(delta[:, 1]))
        
        self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

