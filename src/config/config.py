import torch
import os

# Data configuration
DATA_DIR = 'data'
SAVE_DIR = 'outputs'
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE = 256  # Updated to match decoder output size
BATCH_SIZE = 16
NUM_WORKERS = 4

# Model configuration
MODEL_CONFIG = {
    'NUM_CLASSES': 1,
    'PRETRAINED': True,
    'FREEZE_BACKBONE': True,
    'DROPOUT': 0.1
}

# Training configuration
TRAINING_CONFIG = {
    'NUM_EPOCHS': 100,
    'LEARNING_RATE': 2e-4,
    'WEIGHT_DECAY': 1e-4,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'TRAIN_SIZE': 0.7,
    'VAL_SIZE': 0.15,
    'TEST_SIZE': 0.15,
    'SEED': 42,
    'BCE_WEIGHT': 1.0,
    'DICE_WEIGHT': 1.0,
    'DEEP_SUPERVISION_WEIGHT': 0.4,
    'AUX_WEIGHT': 0.2
} 