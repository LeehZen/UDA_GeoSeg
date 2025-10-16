from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.bhopal_test import *
from geoseg.models.DCSwin import dcswin_base
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
weights_path = "model_weights/mixed_bhopal_potsdam/{}".format(weights_name)
# test_weights_name = "dcswin-base-1024-e30-v1"
test_weights_name = "dcswin-base-1024-e30"
log_name = 'bhopal_test/{}'.format(weights_name)
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

# New hyperparameters for Self-Training
PSEUDO_LABEL_THRESHOLD = 0.95  # Confidence threshold for creating pseudo-labels
LAMBDA_ST = 0.1                # Weight for the self-training loss
LAMBDA_ADV = 0.01     

# Define the optimizer for the discriminator
lr_d = 2e-4 # Learning rate for discriminator
optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.99))

# Define the adversarial loss
adv_loss = torch.nn.BCEWithLogitsLoss()


test_dataset = BhopalDataset()


# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
