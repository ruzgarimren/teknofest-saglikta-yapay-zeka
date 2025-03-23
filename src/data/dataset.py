import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class PolypDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load image and mask paths
        self.images_dir = os.path.join(data_dir, 'PNG', 'Original')
        self.masks_dir = os.path.join(data_dir, 'PNG', 'Ground Truth')
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(self.images_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        # Read image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Apply transformations
        if self.transform:
            image = self.transform['image'](image)
            mask = self.transform['mask'](mask)
        
        return {
            'image': image,
            'mask': mask,
            'image_path': img_path
        }

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