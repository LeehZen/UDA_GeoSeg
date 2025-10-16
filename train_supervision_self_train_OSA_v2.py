import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random

# (seed_everything and get_args remain the same)
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class Supervision_Train_UDA(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.save_hyperparameters()
        
        # Set to manual optimization
        self.automatic_optimization = False

        # Define models and losses
        self.net = config.net
        self.discriminator = config.discriminator
        self.loss = config.loss
        self.adv_loss = config.adv_loss

        # Metrics and labels
        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)
        self.source_label = 1
        self.target_label = 0

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        source_batch = batch['source']
        target_batch = batch['target']
        source_images, source_masks = source_batch['img'], source_batch['gt_semantic_seg']
        target_images = target_batch['img']

        # -----------------------------------------
        #  MANUAL STEP 1: Train Generator (Segmentation Net)
        # -----------------------------------------
        
        # Supervised Segmentation Loss
        pred_source = self.net(source_images)
        loss_seg = self.loss(pred_source, source_masks)
        
        if self.config.use_aux_loss:
            pre_mask = F.softmax(pred_source[0], dim=1).argmax(dim=1)
        else:
            pre_mask = F.softmax(pred_source, dim=1).argmax(dim=1)
        for i in range(source_masks.shape[0]):
            self.metrics_train.add_batch(source_masks[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # Self-Training and Adversarial Loss
        pred_target = self.net(target_images)
        main_pred_target = pred_target[0] if isinstance(pred_target, tuple) else pred_target
        
        # --- Self-Training Logic ---
        with torch.no_grad():
            probabilities = F.softmax(main_pred_target, dim=1)
            confidence, pseudo_labels = torch.max(probabilities, dim=1)
            mask = confidence > self.config.PSEUDO_LABEL_THRESHOLD
        
        loss_st = torch.tensor(0.0, device=self.device)
        if mask.any():
            # Create a new tensor for pseudo-labels, filling unconfident pixels with ignore_index
            masked_pseudo_labels = torch.full_like(pseudo_labels, self.config.ignore_index)
            masked_pseudo_labels[mask] = pseudo_labels[mask]
            
            # Calculate loss using the full-sized masked tensor
            loss_st = self.loss(pred_target, masked_pseudo_labels)

        # Adversarial Loss
        output_target_for_adv = F.softmax(main_pred_target, dim=1)
        disc_preds_on_target = self.discriminator(output_target_for_adv)
        loss_adv = self.adv_loss(disc_preds_on_target, torch.full(disc_preds_on_target.shape, self.source_label, dtype=torch.float, device=self.device))

        # Combine all three generator losses
        g_loss = loss_seg + (self.config.LAMBDA_ADV * loss_adv) + (self.config.LAMBDA_ST * loss_st)
        self.log_dict({'g_loss': g_loss, 'seg_loss': loss_seg, 'st_loss': loss_st}, prog_bar=True)
        
        # Manual backward pass for generator
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # -----------------------------------------
        #  MANUAL STEP 2: Train Discriminator
        # -----------------------------------------
        
        # Train with "real" examples (source)
        main_pred_source = pred_source[0] if isinstance(pred_source, tuple) else pred_source
        output_source = F.softmax(main_pred_source.detach(), dim=1)
        disc_preds_on_source = self.discriminator(output_source)
        loss_disc_source = self.adv_loss(disc_preds_on_source, torch.full(disc_preds_on_source.shape, self.source_label, dtype=torch.float, device=self.device))

        # Train with "fake" examples (target)
        output_target_detached = output_target_for_adv.detach()
        disc_preds_on_target_d = self.discriminator(output_target_detached)
        loss_disc_target = self.adv_loss(disc_preds_on_target_d, torch.full(disc_preds_on_target_d.shape, self.target_label, dtype=torch.float, device=self.device))

        d_loss = (loss_disc_source + loss_disc_target) * 0.5
        self.log('d_loss', d_loss, prog_bar=True)
        
        # Manual backward pass for discriminator
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
    
    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        if schedulers:
            schedulers.step()

        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        self.metrics_train.reset()
        self.log_dict({'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        main_pred = prediction[0] if isinstance(prediction, tuple) else prediction
        loss_val = self.loss(prediction, mask)
        self.log('val_loss', loss_val, on_step=False, on_epoch=True, prog_bar=True)

        pre_mask = F.softmax(main_pred, dim=1).argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

    def on_validation_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())
        
        OA = np.nanmean(self.metrics_val.OA())
        self.metrics_val.reset()
        self.log_dict({'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}, prog_bar=True)

    def configure_optimizers(self):
        opt_g = self.config.optimizer
        opt_d = self.config.optimizer_d
        sch_g = self.config.lr_scheduler
        return [opt_g, opt_d], [sch_g]

    def train_dataloader(self):
        return {'source': self.config.train_loader, 'target': self.config.target_loader}

    def val_dataloader(self):
        return self.config.val_loader

def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Supervision_Train_UDA(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train_UDA.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback],
                         logger=logger)
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)

if __name__ == "__main__":
   main()