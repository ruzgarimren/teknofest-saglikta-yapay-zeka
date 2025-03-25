import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import numpy as np
import segmentation_models_pytorch as smp
from data.dataset import MultiDataset, get_preprocessing
from config.config import TRAINING_CONFIG, MODEL_CONFIG, DATASET_CONFIGS, get_data_config

def dice_loss(pred, target):
    smooth = 1e-5
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

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
            "dataset": dataset_config['name'],
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            **MODEL_CONFIG
        }
    )
    
    # Use wandb for device selection
    device = wandb.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nStarting training on {dataset_config['name']}")
    print(f"Data directory: {data_config['data_dir']}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Using device: {device}")
    print("-" * 50)
    
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
        classes=1
    )
    
    # Log model to wandb
    wandb.watch(model)
    
    model = model.to(device)
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_iou = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_bar = tqdm(train_loader, desc='Training')
        
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            # Calculate training metrics
            pred = torch.sigmoid(outputs)
            pred = (pred > 0.5).float()
            intersection = (pred * masks).sum()
            union = pred.sum() + masks.sum() - intersection
            batch_iou = (intersection + 1e-6) / (union + 1e-6)
            
            train_loss += loss.item()
            train_iou += batch_iou.item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_iou.item():.4f}'
            })
        
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_iou = 0.0
        valid_bar = tqdm(valid_loader, desc='Validation')
        
        with torch.no_grad():
            for images, masks in valid_bar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                pred = torch.sigmoid(outputs)
                pred = (pred > 0.5).float()
                intersection = (pred * masks).sum()
                union = pred.sum() + masks.sum() - intersection
                batch_iou = (intersection + 1e-6) / (union + 1e-6)
                
                valid_loss += loss.item()
                valid_iou += batch_iou.item()
                
                valid_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{batch_iou.item():.4f}'
                })
        
        valid_loss /= len(valid_loader)
        valid_iou /= len(valid_loader)
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/iou": train_iou,
            "val/loss": valid_loss,
            "val/iou": valid_iou,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f"\nEpoch Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {valid_loss:.4f} | Val IoU: {valid_iou:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if valid_iou > best_iou:
            best_iou = valid_iou
            model_path = os.path.join('outputs', f'best_model_dataset_{args.dataset}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model with IoU: {best_iou:.4f}")
        
        scheduler.step()
        print("-" * 50)
    
    wandb.finish()

if __name__ == "__main__":
    main() 