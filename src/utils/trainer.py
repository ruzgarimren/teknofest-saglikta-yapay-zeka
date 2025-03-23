import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def iou_score(pred, target):
    """
    Calculate IoU score between prediction and target
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    return (intersection + 1e-6) / (union + 1e-6)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    """
    Train the segmentation model
    """
    best_iou = 0.0
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_iou = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            epoch_loss += loss.item()
            epoch_iou += iou_score(outputs, masks).item()
            num_batches += 1

        train_loss = epoch_loss / num_batches
        train_iou = epoch_iou / num_batches
        train_losses.append(train_loss)
        train_ious.append(train_iou)

        # Validation phase
        model.eval()
        epoch_loss = 0
        epoch_iou = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                epoch_loss += loss.item()
                epoch_iou += iou_score(outputs, masks).item()
                num_batches += 1

        val_loss = epoch_loss / num_batches
        val_iou = epoch_iou / num_batches
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    return train_losses, val_losses, train_ious, val_ious

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test set
    """
    model.eval()
    test_ious = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            iou = iou_score(outputs, masks).item()
            test_ious.append(iou)

    mean_iou = np.mean(test_ious)
    print(f'\nTest IoU: {mean_iou:.4f}')
    return mean_iou

def plot_training_curves(train_losses, val_losses, train_ious, val_ious, save_dir):
    """
    Plot training curves
    """
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    # Plot IoUs
    plt.subplot(1, 2, 2)
    plt.plot(train_ious, label='Train IoU')
    plt.plot(val_ious, label='Val IoU')
    plt.title('IoU Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def visualize_predictions(model, test_loader, device, num_samples=5, save_dir=None):
    """
    Visualize model predictions
    """
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)
            
            # Convert to numpy arrays
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            
            # Plot results
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(np.transpose(images[0], (1, 2, 0)))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(masks[0, 0], cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(preds[0, 0], cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'prediction_{i+1}.png'))
            plt.close()

def plot_confusion_matrix(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close() 