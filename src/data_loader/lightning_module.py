
import torch
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
            self.init_poses_noise = traning_param['init_pose_max_noise']
            self.loss_w_trans = traning_param['loss_weight_trans']
            self.loss_w_rot = traning_param['loss_weight_rot']
            self.loss_w_proj_r = traning_param['loss_weight_proj_r']
            self.loss_w_proj_theta = traning_param['loss_weight_proj_theta']
                        
        else:
            self.supervised = False
            self.init_poses_noise = traning_param['init_pose_max_noise']
            self.loss_w_proj_r = traning_param['loss_weight_proj_r']
            self.loss_w_proj_theta = traning_param['loss_weight_proj_theta']


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        # shcelduler 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", 
                "frequency": 1,      
                "monitor": "val_loss", 
            },
        }
    
    def training_step(self, batch, batch_idx):

        if self.supervised: 
            # freeze poses 
            if self.global_step < self.freeze_poses_steps:
                freeze_poses = True
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
                        supervised=self.supervised, # Jawne przypisanie
                        freeze_poses=freeze_poses, 
                        init_poses_noise=self.init_poses_noise, 
                        debug_logger=False)

        for k, (pred_poses, target_projection, predicted_projection, valid_mask) in enumerate(pred):
            
            if self.supervised:
                # pose eror - mean from absolute pose and rotation error
                trans_err, rot_err = pose_err(pred_poses, trajectory_gt)
                # accumulate loss components
                loss_trans += trans_err 
                loss_rot += rot_err
            
            # --- projection error --- 
            # supervised - between prediction and gt
            # selfsupervised - between prediction and optimized value by BA
            valid_edges_num = torch.sum(valid_mask) + 1e-6
            patch_proj_err = valid_mask.unsqueeze(-1) * torch.abs(target_projection - predicted_projection)
            proj_x_err = torch.sum(patch_proj_err[:, :, 0]) / valid_edges_num # theta err 
            proj_y_err = torch.sum(patch_proj_err[:, :, 1]) / valid_edges_num # r err

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

            self.log_dict({'loss_translation':loss_trans, 'loss_rotation':loss_rot, 'loss_projection_theta':loss_theta, 'loss_projection_r':loss_r}, on_step=True, on_epoch=True, logger=True)

            total_loss = self.loss_w_trans * loss_trans + \
                         self.loss_w_rot * loss_rot + \
                         self.loss_w_proj_r * loss_r + \
                         self.loss_w_proj_theta * loss_theta
            
        else:

            self.log_dict({'loss_projection_theta':loss_theta, 'loss_projection_r':loss_r}, on_step=True, on_epoch=True, logger=True)

            total_loss = self.loss_w_proj_r * loss_r + \
                         self.loss_w_proj_theta * loss_theta

        self.log('total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_loss
        

    def validation_step(self, batch, batch_idx):
        
        # freeze poses 
        if self.supervised:
            freeze_poses = False

        fls_series, time, trajectory_gt, depth_gt = batch

        pred = self.model(frames=fls_series, 
                        timestamp=time, 
                        poses_gt=trajectory_gt, 
                        depth_gt=depth_gt, 
                        supervised=self.supervised, 
                        freeze_poses=freeze_poses, 
                        init_poses_noise=self.init_poses_noise, 
                        debug_logger=False)

        pred_poses, target_projection, predicted_projection, valid_mask = pred[-1]
        
        valid_edges_num = torch.sum(valid_mask) + 1e-6
        patch_proj_err = valid_mask.unsqueeze(-1) * torch.abs(target_projection - predicted_projection)
        proj_x_err = torch.sum(patch_proj_err[:, 0], dim=-1) / valid_edges_num
        proj_y_err = torch.sum(patch_proj_err[:, 1], dim=-1) / valid_edges_num


        metrics = eval_metrics(pred_poses.detach().cpu().numpy(), 
                               trajectory_gt.detach().cpu().numpy(),
                               align=False, align_init_pt_only=True, add_data_series=False)
        
        metrics['projection_err_theta'] = proj_x_err
        metrics['projection_err_r'] = proj_y_err

        self.log_dict(metrics, on_step=True, on_epoch=True, logger=True)

        self.log('val_loss', metrics['MEAN_TRANS_ERR'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        
        
