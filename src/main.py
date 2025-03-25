import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import segmentation_models_pytorch as smp
from data.dataset import MultiDataset, get_preprocessing
from config.config import TRAINING_CONFIG, MODEL_CONFIG, DATA_CONFIG

def dice_loss(pred, target):
    smooth = 1e-5
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def main():
    torch.manual_seed(TRAINING_CONFIG['SEED'])
    
    dataset_id = int(input("Select dataset (1, 2, or 3): "))
    if dataset_id not in [1, 2, 3]:
        raise ValueError("Dataset ID must be 1, 2, or 3")
    
    wandb.init(
        project="polyp-segmentation",
        config={
            "architecture": "DeepLabV3+",
            "dataset": f"Dataset_{dataset_id}",
            **MODEL_CONFIG,
            **TRAINING_CONFIG
        }
    )
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(MODEL_CONFIG['ENCODER'], MODEL_CONFIG['ENCODER_WEIGHTS'])
    preprocessing = get_preprocessing(preprocessing_fn)
    
    train_dataset = MultiDataset(
        dataset_id=dataset_id,
        data_dir=DATA_CONFIG['data_dir'],
        train=True,
        preprocessing=preprocessing
    )
    
    valid_dataset = MultiDataset(
        dataset_id=dataset_id,
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
    device = TRAINING_CONFIG['DEVICE']
    model = model.to(device)
    
    for epoch in range(TRAINING_CONFIG['NUM_EPOCHS']):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        model.eval()
        valid_loss = 0.0
        iou = 0.0
        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                valid_loss += loss.item()
                
                pred = torch.sigmoid(outputs)
                pred = (pred > 0.5).float()
                intersection = (pred * masks).sum()
                union = pred.sum() + masks.sum() - intersection
                iou += (intersection + 1e-6) / (union + 1e-6)
        
        valid_loss /= len(valid_loader)
        iou /= len(valid_loader)
        
        wandb.log({
            "train/loss": train_loss,
            "val/loss": valid_loss,
            "val/iou": iou,
            "epoch": epoch
        })
        
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), os.path.join(TRAINING_CONFIG['output_dir'], f'best_model_dataset_{dataset_id}.pth'))
        
        print(f'Epoch {epoch+1}/{TRAINING_CONFIG["NUM_EPOCHS"]}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {valid_loss:.4f}')
        print(f'IoU: {iou:.4f}')
        print('-' * 50)

if __name__ == '__main__':
    main()