import torch
import os

# Data configuration
DATA_DIR = 'data'
SAVE_DIR = 'outputs'
os.makedirs(SAVE_DIR, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'ENCODER': 'resnet50',
    'ENCODER_WEIGHTS': 'imagenet',
    'NUM_CLASSES': 1,
    'ACTIVATION': 'sigmoid'
}

# Training configuration
TRAINING_CONFIG = {
    'SEED': 42,
    'NUM_EPOCHS': 15,
    'BATCH_SIZE': 8,
    'LEARNING_RATE': 0.00008,
    'TRAIN_SIZE': 0.7,
    'VAL_SIZE': 0.15,
    'NUM_WORKERS': 4,
    'output_dir': 'outputs'
}

# Data Configuration
DATA_CONFIG = {
    'data_dir': 'data'
}

# Create output directory if it doesn't exist
os.makedirs(TRAINING_CONFIG['output_dir'], exist_ok=True)

# Training configuration
TRAINING_CONFIG = {
    'NUM_EPOCHS': 15,
    'LEARNING_RATE': 0.00008,
    'WEIGHT_DECAY': 1e-4,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'TRAIN_SIZE': 0.9,
    'VAL_SIZE': 0.1,
    'TEST_SIZE': 0.0,
    'SEED': 42,
    'BATCH_SIZE': 16,
    'NUM_WORKERS': 4
} 