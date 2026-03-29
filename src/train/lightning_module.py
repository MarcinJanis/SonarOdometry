
import torch
import lightning.pytorch as pl

from .utils import pose_err


class DPSO_LightningModule(pl.LightningModule):
    def __init__(self, model, freeze_poses_step, init_poses_noise, loss_weights, opt_iter_weights):
        super().__init__()

        self.model = model

        self.save_hyperparameters()

        self.freeze_poses_step = freeze_poses_step

        self.loss_w = loss_weights # weights of loss components
        self.opt_iter_weights = opt_iter_weights # weights of loss for each optim iteration

        self.init_poses_noise  = init_poses_noise # noise for initial poses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # shcelduler 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # lub "step" - jak często aktualizować
                "frequency": 1,      # co ile interwałów (np. co każdą epokę)
                "monitor": "val_loss", # wymagane np. dla ReduceLROnPlateau
            },
        }
    
    def training_step(self, batch, batch_idx):

        # freeze poses 
        if self.global_step < self.freeze_poses_step:
            freeze_poses = True
        else:
            freeze_poses = False

        loss_trans = 0.0
        loss_rot = 0.0
        loss_theta = 0.0
        loss_r = 0.0

        fls_series, time, trajectory_gt, depth_gt = batch

        pred = self.model(fls_series, time, trajectory_gt, depth_gt, 
                          freeze_poses=freeze_poses, 
                          init_poses_noise = self.init_poses_noise, 
                          debug_logger=False)

        for k, (pred_poses, pred_coords, gt_coords) in enumerate(pred):
            
            trans_err, rot_err = pose_err(pred_poses, trajectory_gt)

            patch_proj_err = torch.abs(gt_coords - pred_coords)
            proj_x_err = torch.mean(patch_proj_err[:, 0], dim=-1)
            proj_y_err = torch.mean(patch_proj_err[:, 1], dim=-1)

            # accumulate loss
            loss_trans += trans_err * self.opt_iter_weights[k]
            loss_rot += rot_err * self.opt_iter_weights[k]
            loss_theta += proj_x_err * self.opt_iter_weights[k]
            loss_r += proj_y_err * self.opt_iter_weights[k]

        # compute loss 
        loss_trans = loss_trans / k
        loss_rot = loss_rot / k
        loss_theta = loss_theta / k
        loss_r = loss_r / k

        # log loss components
        self.log_dict({'loss_trans':loss_trans, 'loss_rot':loss_rot, 'loss_proj_theta':loss_theta, 'loss_proj_r':loss_r}, on_step=True, on_epoch=True, logger=True)

        # total loss
        loss = loss_trans*self.loss_w['trans'] + \
               loss_rot*self.loss_w['rot'] + \
               loss_theta*self.loss_w['proj_theta'] + \
               loss_r*self.loss_w['proj_r']

        self.log('total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
        

    def validation_step(self, batch, batch_idx):
        
        # freeze poses 
        freeze_poses = False

        loss_trans = 0.0
        loss_rot = 0.0
        loss_theta = 0.0
        loss_r = 0.0

        fls_series, time, trajectory_gt, depth_gt = batch

        pred = self.model(fls_series, time, trajectory_gt, depth_gt, 
                          freeze_poses=freeze_poses, 
                          init_poses_noise = self.init_poses_noise, 
                          debug_logger=False)

        # for k, (pred_poses, pred_coords, gt_coords) in enumerate(pred):
        pred_poses, pred_coords, gt_coords = pred[-1]

        trans_err, rot_err = pose_err(pred_poses, trajectory_gt)

        patch_proj_err = torch.abs(gt_coords - pred_coords)
        proj_x_err = torch.mean(patch_proj_err[:, 0], dim=-1)
        proj_y_err = torch.mean(patch_proj_err[:, 1], dim=-1)

        # loss components
        loss_trans = trans_err 
        loss_rot = rot_err 
        loss_theta = proj_x_err 
        loss_r  =  proj_y_err

        # total loss
        loss = loss_trans*self.loss_w['trans'] + \
               loss_rot*self.loss_w['rot'] + \
               loss_theta*self.loss_w['proj_theta'] + \
               loss_r*self.loss_w['proj_r']

        # log loss components
        self.log_dict({'loss_trans':loss_trans, 'loss_rot':loss_rot, 'loss_proj_theta':loss_theta, 'loss_proj_r':loss_r}, on_step=True, on_epoch=True, logger=True)

        self.log('total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        
        
