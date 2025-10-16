import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random


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
        self.net = config.net

        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

        self.discriminator = config.disc
        self.adv_loss = config.adv_loss # disc loss
        
        self.source_label = 1
        self.target_label = 0

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx, optimizer_idx):
        # The batch now contains data from both source and target dataloaders
        source_batch = batch['source']
        target_batch = batch['target']

        source_images, source_masks = source_batch['img'], source_batch['gt_semantic_seg']
        target_images = target_batch['img']

        # -----------------------------------------
        #  Train Generator (Segmentation Network)
        # -----------------------------------------
        if optimizer_idx == 0:
            # 1. Supervised Segmentation Loss (on labeled source data)
            # Your original loss calculation
            pred_source = self.net(source_images)
            loss_seg = self.loss(pred_source, source_masks)

            # Update training metrics with source data predictions
            if self.config.use_aux_loss:
                pre_mask = nn.Softmax(dim=1)(pred_source[0])
            else:
                pre_mask = nn.Softmax(dim=1)(pred_source)
            pre_mask = pre_mask.argmax(dim=1)
            for i in range(source_masks.shape[0]):
                self.metrics_train.add_batch(source_masks[i].cpu().numpy(), pre_mask[i].cpu().numpy())

            # 2. Adversarial Loss (on unlabeled target data)
            # We want the generator to fool the discriminator
            pred_target = self.net(target_images)
            # The output of the segmentation net might be a tuple (main_out, aux_out)
            main_pred_target = pred_target[0] if isinstance(pred_target, tuple) else pred_target
            
            output_target = F.softmax(main_pred_target, dim=1)
            disc_preds_on_target = self.discriminator(output_target)

            # The generator's goal is to make the discriminator think the target predictions are from the source (label=1)
            loss_adv = self.adv_loss(disc_preds_on_target, torch.full(disc_preds_on_target.shape, self.source_label, dtype=torch.float, device=self.device))

            # Combine generator losses (you can add a weight, e.g., 0.01, to the adversarial loss)
            g_loss = loss_seg + 0.01 * loss_adv
            self.log('g_loss', g_loss, prog_bar=True, logger=True)
            self.log('seg_loss', loss_seg, logger=True)
            
            return g_loss

        # -----------------------------------------
        #  Train Discriminator
        # -----------------------------------------
        if optimizer_idx == 1:
            # 1. Train with "real" examples (predictions from source data)
            pred_source = self.net(source_images)
            main_pred_source = pred_source[0] if isinstance(pred_source, tuple) else pred_source
            output_source = F.softmax(main_pred_source.detach(), dim=1)
            
            disc_preds_on_source = self.discriminator(output_source)
            loss_disc_source = self.adv_loss(disc_preds_on_source, torch.full(disc_preds_on_source.shape, self.source_label, dtype=torch.float, device=self.device))

            # --- METRIC CALCULATION ---
            # Apply sigmoid to get probabilities and round to get predictions (0 or 1)
            acc_source = (torch.sigmoid(disc_preds_on_source).round() == self.source_label).float().mean()
            self.log('d_acc_source', acc_source, prog_bar=True, logger=True)
            # --- END METRIC ---

            # 2. Train with "fake" examples (predictions from target data)
            pred_target = self.net(target_images)
            main_pred_target = pred_target[0] if isinstance(pred_target, tuple) else pred_target
            output_target = F.softmax(main_pred_target.detach(), dim=1)
            
            disc_preds_on_target = self.discriminator(output_target)
            loss_disc_target = self.adv_loss(disc_preds_on_target, torch.full(disc_preds_on_target.shape, self.target_label, dtype=torch.float, device=self.device))

            # --- METRIC CALCULATION ---
            acc_target = (torch.sigmoid(disc_preds_on_target).round() == self.target_label).float().mean()
            self.log('d_acc_target', acc_target, prog_bar=True, logger=True)
            # --- END METRIC ---

            # Backward pass for discriminator losses
            d_loss = (loss_disc_source + loss_disc_target) * 0.5
            self.log('d_loss', d_loss, prog_bar=True, logger=True)
            
        return d_loss

    def on_train_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('train:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # 'batch' here comes ONLY from your val_loader, which should be your target domain's validation set.
        img, mask = batch['img'], batch['gt_semantic_seg']

        # The rest of your logic is correct...
        prediction = self.forward(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        self.log('val_loss', loss_val)

    def on_validation_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        opt_d = self.config.optimizer_d # discriminator optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer, opt_d], [lr_scheduler]

    def train_dataloader(self):

        return {
            'source': self.config.train_loader,
            'target': self.config.target_loader
        }

    def val_dataloader(self):

        return self.config.val_loader


# training
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
                         callbacks=[checkpoint_callback], strategy='auto',
                         logger=logger)
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
   main()