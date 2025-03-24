import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import segmentation_models_pytorch as smp
from data.dataset import PolypDataset, get_preprocessing
from config.config import TRAINING_CONFIG, MODEL_CONFIG, DATA_CONFIG

def dice_loss(pred, target):
    smooth = 1e-5
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def calculate_iou(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).bool()
    target = target.bool()
    intersection = (pred & target).sum((2,3))
    union = (pred | target).sum((2,3))
    iou = (intersection.float() / union.float()).mean()
    return iou

def main():
    # Set random seed
    torch.manual_seed(TRAINING_CONFIG['SEED'])
    
    # Initialize wandb
    wandb.init(
        project="polyp-segmentation",
        config={
            "architecture": "DeepLabV3+",
            "dataset": "CVC-ClinicDB",
            **MODEL_CONFIG,
            **TRAINING_CONFIG
        }
    )
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(MODEL_CONFIG['ENCODER'], MODEL_CONFIG['ENCODER_WEIGHTS'])
    preprocessing = get_preprocessing(preprocessing_fn)
    
    train_dataset = PolypDataset(
        data_dir=DATA_CONFIG['data_dir'],
        train=True,
        preprocessing=preprocessing
    )
    
    valid_dataset = PolypDataset(
        data_dir=DATA_CONFIG['data_dir'],
        train=False,
        preprocessing=preprocessing
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=TRAINING_CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = smp.DeepLabV3Plus(
        encoder_name=MODEL_CONFIG['ENCODER'],
        encoder_weights=MODEL_CONFIG['ENCODER_WEIGHTS'],
        in_channels=3,
        classes=1
    )
    
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG['LEARNING_RATE'])
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAINING_CONFIG['NUM_EPOCHS'])
    
    best_iou = 0.0
    for epoch in range(TRAINING_CONFIG['NUM_EPOCHS']):
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            batch_iou = calculate_iou(outputs, masks)
            train_loss += loss.item()
            train_iou += batch_iou.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{TRAINING_CONFIG["NUM_EPOCHS"]}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, IoU: {batch_iou.item():.4f}')
        
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        model.eval()
        valid_loss = 0.0
        valid_iou = 0.0
        with torch.no_grad():
            for images, masks in valid_loader:
                outputs = model(images)
                loss = criterion(outputs, masks)
                valid_loss += loss.item()
                valid_iou += calculate_iou(outputs, masks).item()
        
        valid_loss /= len(valid_loader)
        valid_iou /= len(valid_loader)
        
        print(f'Epoch {epoch+1}/{TRAINING_CONFIG["NUM_EPOCHS"]}:')
        print(f'Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid IoU: {valid_iou:.4f}')
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": valid_loss,
            "val_iou": valid_iou
        })
        
        if valid_iou > best_iou:
            best_iou = valid_iou
            torch.save(model.state_dict(), os.path.join(TRAINING_CONFIG['output_dir'], 'best_model.pth'))
            print(f'New best model saved with IoU: {best_iou:.4f}')
        
        scheduler.step()
    
    wandb.finish()

if __name__ == '__main__':
    main()