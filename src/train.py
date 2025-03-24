import os
import cv2
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils

def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map

def reverse_one_hot(image):
    x = np.argmax(image, axis = -1)
    return x

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x

class EndoscopyDataset(torch.utils.data.Dataset):
    def __init__(self, df, class_rgb_values=None, augmentation=None, preprocessing=None):
        self.image_paths = df['png_image_path'].tolist()
        self.mask_paths = df['png_mask_path'].tolist()
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.image_paths)

def get_training_augmentation():
    train_transform = [
        album.HorizontalFlip(p=0.5),
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        album.PadIfNeeded(min_height=288, min_width=384, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)

def crop_image(image, true_dimensions):
    return album.CenterCrop(p=1, height=true_dimensions[0], width=true_dimensions[1])(image=image)

def main():
    wandb.init(project="polyp-segmentation", config={
        "encoder": "resnet50",
        "encoder_weights": "imagenet",
        "classes": ["background", "polyp"],
        "activation": "sigmoid",
        "epochs": 15,
        "batch_size": 16,
        "learning_rate": 0.00008,
        "optimizer": "adam",
        "loss": "dice_loss"
    })

    DATA_DIR = 'data/dataset'
    metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
    metadata_df = metadata_df[['frame_id', 'png_image_path', 'png_mask_path']]
    metadata_df['png_image_path'] = metadata_df['png_image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata_df['png_mask_path'] = metadata_df['png_mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

    valid_df = metadata_df.sample(frac=0.1, random_state=42)
    train_df = metadata_df.drop(valid_df.index)

    class_dict = pd.read_csv(os.path.join(DATA_DIR, 'class_dict.csv'))
    class_names = class_dict['class_names'].tolist()
    class_rgb_values = class_dict[['r','g','b']].values.tolist()

    select_classes = ['background', 'polyp']
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    model = smp.DeepLabV3Plus(
        encoder_name=wandb.config.encoder,
        encoder_weights=wandb.config.encoder_weights,
        classes=len(wandb.config.classes),
        activation=wandb.config.activation,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(wandb.config.encoder, wandb.config.encoder_weights)

    train_dataset = EndoscopyDataset(
        train_df,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    valid_dataset = EndoscopyDataset(
        valid_df,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=0)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)

    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=wandb.config.learning_rate)])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, wandb.config.epochs):
        print(f'\nEpoch: {i}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        wandb.log({
            "train/dice_loss": train_logs['dice_loss'],
            "train/iou_score": train_logs['iou_score'],
            "val/dice_loss": valid_logs['dice_loss'],
            "val/iou_score": valid_logs['iou_score'],
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, 'outputs/best_model.pth')
            print('Model saved!')

    wandb.finish()

if __name__ == "__main__":
    main() 