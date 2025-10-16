from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.mixed_dataset import *
from geoseg.datasets.bhopal_test import *
from geoseg.models.DCSwin import dcswin_small, dcswin_base
from tools.utils import Lookahead
from tools.utils import process_model_params
from timm.scheduler.poly_lr import PolyLRScheduler
from geoseg.models.output_discriminator import OutputDiscriminator

# training hparam
max_epoch = 30
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 4
lr = 1e-3
weight_decay = 2.5e-4
backbone_lr = 1e-4
backbone_weight_decay = 2.5e-4
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "dcswin-base-1024-e30"
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "dcswin-base-1024-e30"
log_name = 'mixed_bhopal_potsdam/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = dcswin_base(num_classes=num_classes)
discriminator = OutputDiscriminator(num_classes=num_classes)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
# loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = False

# define the dataloader
def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


train_dataset = MixedDataset(data_root='data/mixed_bhopal_potsdam/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = MixedDataset(transform=val_aug)
test_dataset = MixedDataset(data_root='data/mixed_bhopal_potsdam/test',
                              transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
# base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
# optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


# Define the optimizer for the discriminator
lr_d = 2e-4 # Learning rate for discriminator
optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.99))

# Define the adversarial loss
adv_loss = torch.nn.BCEWithLogitsLoss()

target_dataset = BhopalDataset()

target_loader = DataLoader(dataset=target_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

# New hyperparameters for Self-Training
PSEUDO_LABEL_THRESHOLD = 0.95  # Confidence threshold for creating pseudo-labels
LAMBDA_ST = 0.1                # Weight for the self-training loss
LAMBDA_ADV = 0.01                          