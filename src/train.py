import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from data.dataset import MultiDataset, get_preprocessing
from config.config import TRAINING_CONFIG, MODEL_CONFIG, DATASET_CONFIGS, get_data_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--dataset', type=int, choices=[1, 2, 3], required=True,
                      help='Dataset ID to train on (1, 2, or 3)')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.00008,
                      help='Learning rate')
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_config = DATASET_CONFIGS[args.dataset]
    data_config = get_data_config(args.dataset)
    torch.manual_seed(TRAINING_CONFIG['SEED'])
    
    wandb.init(
        project="polyp-segmentation",
        config={
            "architecture": "DeepLabV3+",
            "encoder": MODEL_CONFIG['ENCODER'],
            "encoder_weights": MODEL_CONFIG['ENCODER_WEIGHTS'],
            "classes": ["background", "polyp"],
            "activation": MODEL_CONFIG['ACTIVATION'],
            "dataset": dataset_config['name'],
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "optimizer": "adam",
            "loss": "dice_loss"
        }
    )
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(MODEL_CONFIG['ENCODER'], MODEL_CONFIG['ENCODER_WEIGHTS'])
    preprocessing = get_preprocessing(preprocessing_fn)
    
    train_dataset = MultiDataset(
        dataset_id=args.dataset,
        data_dir=data_config['data_dir'],
        train=True,
        preprocessing=preprocessing
    )
    
    valid_dataset = MultiDataset(
        dataset_id=args.dataset,
        data_dir=data_config['data_dir'],
        train=False,
        preprocessing=preprocessing
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print("-" * 50)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    model = smp.DeepLabV3Plus(
        encoder_name=MODEL_CONFIG['ENCODER'],
        encoder_weights=MODEL_CONFIG['ENCODER_WEIGHTS'],
        in_channels=3,
        classes=1,
        activation=MODEL_CONFIG['ACTIVATION']
    )
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    
    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=args.learning_rate)])
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
    
    for i in range(args.epochs):
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
            torch.save(model, os.path.join('outputs', f'best_model_dataset_{args.dataset}.pth'))
            print('Model saved!')
        
        lr_scheduler.step()
    
    wandb.finish()

if __name__ == "__main__":
    main() 