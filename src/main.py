import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.unet import UNet
from data.dataset import PolypDataset, get_transforms
from utils.trainer import (
    train_model, evaluate_model, plot_training_curves,
    visualize_predictions
)
from config.config import (
    DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    IN_CHANNELS, OUT_CHANNELS, LEARNING_RATE, NUM_EPOCHS,
    DEVICE, SAVE_DIR
)

def main():
    # Create output directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Setup data
    transform = get_transforms(IMAGE_SIZE)
    dataset = PolypDataset(DATA_DIR, transform=transform)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Setup model
    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    train_losses, val_losses, train_ious, val_ious = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=NUM_EPOCHS, device=DEVICE, save_dir=SAVE_DIR
    )

    # Load best model and evaluate
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth')))
    test_iou = evaluate_model(model, test_loader, DEVICE)

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, train_ious, val_ious,
        save_dir=SAVE_DIR
    )

    # Visualize some predictions
    visualize_predictions(
        model, test_loader, DEVICE,
        num_samples=5, save_dir=SAVE_DIR
    )

if __name__ == '__main__':
    main() 