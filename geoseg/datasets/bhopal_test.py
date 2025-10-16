import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
from .transform import *
import matplotlib.patches as mpatches
from PIL import Image
import random
import rasterio


CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 204, 0], [255, 0, 0]]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)

def get_training_transform():
    train_transform = [
        # albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.15),
        # albu.RandomRotate90(p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=768, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
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

def val_aug(img, mask=None):
    img = np.array(img)
    if mask is not None:
        mask = np.array(mask)
        aug = get_val_transform()(image=img.copy(), mask=mask.copy())
        return aug['image'], aug['mask']
    else:
        aug = get_val_transform()(image=img.copy())
        return aug['image']


class BhopalDataset(Dataset): # only made for testing
    def __init__(self, data_root='data/bhopal_test/test', mode='test', img_dir='images_1024', img_suffix='.tif', transform = val_aug):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mode = mode
        self.img_suffix = img_suffix
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir)
        self.transform = transform

    def __getitem__(self, index):
        img = self.load_img(index)
        if self.transform:
            img = self.transform(img)
        # convert to torch tensor, CHW
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img_id = self.img_ids[index]
        return dict(img_id=img_id, img=img)

    def __len__(self):
        return len(self.img_ids)
    
    def get_img_ids(self, data_root, img_dir):
        img_path = os.path.join(data_root, img_dir)
        img_filename_list = os.listdir(img_path)
        img_ids = [str(id.split('.')[0]) for id in img_filename_list]
        return img_ids

    def load_img(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)

        if self.img_suffix.lower() == ".tif":
            with rasterio.open(img_name) as src:
                # print("Src count:", src.count)
                if src.count < 3:
                    raise ValueError(f"TIF {img_name} has less than 3 channels.")
                # Read first 3 bands (C, H, W)
                arr = src.read([1, 2, 3])
                # Convert to (H, W, C)
                arr = np.transpose(arr, (1, 2, 0))
                # Scale/convert if dtype is not uint8
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                # print(arr.shape)
                img = arr  # keep as numpy array
        else:
            img = np.array(Image.open(img_name).convert("RGB"))

        return img

