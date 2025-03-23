import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

class PolypDataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.images_dir = os.path.join(data_dir, 'PNG', 'Original')
        self.masks_dir = os.path.join(data_dir, 'PNG', 'Ground Truth')
        self.train = train
        
        self.images = sorted([f for f in os.listdir(self.images_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if train:
            self.transform = A.Compose([
                A.RandomResizedCrop(256, 256, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, 
                    scale_limit=0.2, 
                    rotate_limit=45, 
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(
                        alpha=120, 
                        sigma=120 * 0.05, 
                        alpha_affine=120 * 0.03, 
                        p=0.5
                    ),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(
                        distort_limit=1, 
                        shift_limit=0.5, 
                        p=0.5
                    ),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                ], p=0.3),
                A.CoarseDropout(
                    max_holes=8, 
                    max_height=20, 
                    max_width=20, 
                    min_holes=5, 
                    min_height=15, 
                    min_width=15, 
                    fill_value=0, 
                    p=0.3
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        
        mask = torch.unsqueeze((mask > 128).float(), 0)
        return image, mask

def get_transforms(image_size=256):
    """
    Get transforms for both image and mask
    """
    from torchvision import transforms
    
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    return {
        'image': image_transform,
        'mask': mask_transform
    }

def load_data(data_dir):
    image_paths = []
    labels = []
    class_to_idx = {
        'İnme Yok_kronik süreç_diğer Veri Set_PNG': 0,
        'İskemi Veri Seti': 1,
        'Kanama Veri Seti': 2
    }

    for class_name in class_to_idx.keys():
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(class_to_idx[class_name])

    return image_paths, labels 