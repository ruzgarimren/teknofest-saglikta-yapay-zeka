import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as album
from albumentations.pytorch import ToTensorV2
import pandas as pd

def get_training_augmentation():
    train_transform = [
        album.Resize(384, 480, always_apply=True),
        album.HorizontalFlip(p=0.5)
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        album.Resize(384, 480, always_apply=True),
    ]
    return album.Compose(test_transform)

def to_tensor(x, **kwargs):
    if len(x.shape) == 2:
        # For masks (single channel)
        return np.expand_dims(x, axis=0).astype('float32')
    else:
        # For RGB images
        return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    return album.Compose([
        album.Lambda(image=preprocessing_fn),
        album.Lambda(image=to_tensor, mask=to_tensor),
    ])

class MultiDataset(Dataset):
    def __init__(self, dataset_id, data_dir, train=True, preprocessing=None):
        self.dataset_id = dataset_id
        self.data_dir = data_dir
        self.train = train
        self.transform = get_training_augmentation() if train else get_validation_augmentation()
        self.preprocessing = preprocessing
        
        if dataset_id == 1:
            self.images_dir = os.path.join(data_dir, 'PNG', 'Original')
            self.masks_dir = os.path.join(data_dir, 'PNG', 'Ground Truth')
            self.image_files = [f for f in os.listdir(self.images_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        elif dataset_id == 2:
            self.images_dir = os.path.join(data_dir, 'images')
            self.masks_dir = os.path.join(data_dir, 'masks')
            self.image_files = [f for f in os.listdir(self.images_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        elif dataset_id == 3:
            # For dataset 3, we need to collect images from all seq folders
            self.image_paths = []  # Store full paths
            self.mask_paths = []   # Store full paths
            
            # Get all sequence folders (seq1, seq2, etc.)
            seq_folders = [f for f in os.listdir(data_dir) if f.startswith('seq')]
            
            for seq in seq_folders:
                seq_path = os.path.join(data_dir, seq)
                images_dir = os.path.join(seq_path, 'images')
                masks_dir = os.path.join(seq_path, 'masks')
                
                if os.path.exists(images_dir) and os.path.exists(masks_dir):
                    # Get all images in this sequence
                    for img_name in os.listdir(images_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(images_dir, img_name)
                            mask_path = os.path.join(masks_dir, img_name)
                            
                            # Only add if both image and mask exist
                            if os.path.exists(mask_path):
                                self.image_paths.append(img_path)
                                self.mask_paths.append(mask_path)
        else:
            raise ValueError(f"Dataset ID {dataset_id} not supported")
        
    def __len__(self):
        if self.dataset_id == 3:
            return len(self.image_paths)
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if self.dataset_id == 3:
            # For dataset 3, use stored full paths
            img_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]
        else:
            # For dataset 1 and 2, construct paths
            img_name = self.image_files[idx]
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        
        if self.preprocessing:
            transformed = self.preprocessing(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        
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