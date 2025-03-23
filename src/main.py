import os
import torch
from torch.utils.data import DataLoader, random_split
import wandb

from models.swin_birefnet import SwinBirefNet, BCEDiceLoss, iou_score
from data.dataset import PolypDataset
from config.config import *

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_iou = 0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        if isinstance(outputs, tuple):
            main_out, deep3, deep2, aux = outputs
            
            # Calculate losses
            main_loss = criterion(main_out, masks)
            deep3_loss = criterion(deep3, masks)
            deep2_loss = criterion(deep2, masks)
            
            # Calculate mean presence of polyp in each image
            mask_presence = (masks.view(masks.size(0), -1).mean(1) > 0.1).float()
            aux_loss = torch.nn.BCEWithLogitsLoss()(aux.squeeze(), mask_presence)
            
            # Combine losses
            loss = (main_loss + 
                   TRAINING_CONFIG['DEEP_SUPERVISION_WEIGHT'] * (deep3_loss + deep2_loss) / 2 +
                   TRAINING_CONFIG['AUX_WEIGHT'] * aux_loss)
            
            # Calculate IoU for main output
            batch_iou = iou_score(main_out, masks)
        else:
            loss = criterion(outputs, masks)
            batch_iou = iou_score(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_iou += batch_iou.item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, IoU: {batch_iou.item():.4f}')
            wandb.log({
                "batch_loss": loss.item(),
                "batch_iou": batch_iou.item(),
                "learning_rate": optimizer.param_groups[0]['lr']
            })
    
    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    return avg_loss, avg_iou

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, masks)
            batch_iou = iou_score(outputs, masks)
            
            total_loss += loss.item()
            total_iou += batch_iou.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    return avg_loss, avg_iou

def main():
    # Set random seed
    torch.manual_seed(TRAINING_CONFIG['SEED'])
    
    # Initialize wandb
    wandb.init(
        project="polyp-segmentation",
        config={
            "architecture": "Swin-BirefNet",
            "dataset": "CVC-ClinicDB",
            **MODEL_CONFIG,
            **TRAINING_CONFIG
        }
    )
    
    # Create datasets
    dataset = PolypDataset(DATA_DIR, train=True)
    total_size = len(dataset)
    train_size = int(TRAINING_CONFIG['TRAIN_SIZE'] * total_size)
    val_size = int(TRAINING_CONFIG['VAL_SIZE'] * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    model = SwinBirefNet(
        num_classes=MODEL_CONFIG['NUM_CLASSES'],
        pretrained=MODEL_CONFIG['PRETRAINED']
    ).to(TRAINING_CONFIG['DEVICE'])
    
    # Loss and optimizer
    criterion = BCEDiceLoss(
        weights=[TRAINING_CONFIG['BCE_WEIGHT'], TRAINING_CONFIG['DICE_WEIGHT']]
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['LEARNING_RATE'],
        weight_decay=TRAINING_CONFIG['WEIGHT_DECAY']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Training loop
    best_val_iou = 0
    
    print("Starting training...")
    for epoch in range(TRAINING_CONFIG['NUM_EPOCHS']):
        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer,
            TRAINING_CONFIG['DEVICE'], epoch
        )
        
        # Validate
        val_loss, val_iou = validate(
            model, val_loader, criterion,
            TRAINING_CONFIG['DEVICE']
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou
        })
        
        print(f"Epoch {epoch+1}/{TRAINING_CONFIG['NUM_EPOCHS']}")
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_iou': val_iou,
            }, os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"New best model saved! Val IoU: {val_iou:.4f}")
    
    # Test best model
    print("\nLoading best model for testing...")
    checkpoint = torch.load(os.path.join(SAVE_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_iou = validate(
        model, test_loader, criterion,
        TRAINING_CONFIG['DEVICE']
    )
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test IoU: {test_iou:.4f}")
    
    wandb.log({
        "test_loss": test_loss,
        "test_iou": test_iou
    })
    
    wandb.finish()

if __name__ == "__main__":
    main() 