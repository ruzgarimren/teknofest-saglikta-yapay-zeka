import torch
import os

# Data configuration
SAVE_DIR = 'outputs'
os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset configurations
DATASET_CONFIGS = {
    1: {
        'name': 'Dataset_1',
        'base_dir': 'data/dataset',
        'image_dir': os.path.join('data/dataset', 'PNG', 'Original'),
        'mask_dir': os.path.join('data/dataset', 'PNG', 'Ground Truth'),
        'image_size': (384, 480)
    },
    2: {
        'name': 'Dataset_2',
        'base_dir': 'data_2/dataset_2/Kvasir-SEG/Kvasir-SEG',
        'image_dir': os.path.join('data_2/dataset_2/Kvasir-SEG/Kvasir-SEG', 'annotated_images'),
        'mask_dir': os.path.join('data_2/dataset_2/Kvasir-SEG/Kvasir-SEG', 'masks'),
        'image_size': (384, 480)
    },
    3: {
        'name': 'Dataset_3',
        'base_dir': 'data_3/dataset_3',
        'image_dir': os.path.join('data_3/dataset_3', 'images'),
        'mask_dir': os.path.join('data_3/dataset_3', 'masks'),
        'image_size': (384, 480)
    }
}

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
    'WEIGHT_DECAY': 1e-4,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'TRAIN_SIZE': 0.9,
    'VAL_SIZE': 0.1,
    'TEST_SIZE': 0.0,
    'NUM_WORKERS': 4,
    'output_dir': 'outputs'
}

def get_data_config(dataset_id):
    if dataset_id not in DATASET_CONFIGS:
        raise ValueError(f"Dataset ID {dataset_id} not found in configurations")
    return {
        'data_dir': DATASET_CONFIGS[dataset_id]['base_dir'],
        'image_dir': DATASET_CONFIGS[dataset_id]['image_dir'],
        'mask_dir': DATASET_CONFIGS[dataset_id]['mask_dir']
    }

# Create output directory if it doesn't exist
os.makedirs(TRAINING_CONFIG['output_dir'], exist_ok=True) 