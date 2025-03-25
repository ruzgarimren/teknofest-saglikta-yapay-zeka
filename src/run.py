import os
import subprocess
import argparse
import wandb
from config.config import DATASET_CONFIGS

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate segmentation model on multiple datasets')
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
    
    # Verify dataset path exists
    dataset_config = DATASET_CONFIGS[args.dataset]
    dataset_path = dataset_config['base_dir']
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Create necessary directories
    os.makedirs('outputs', exist_ok=True)
    
    # Run training with dataset selection
    print(f"Starting training on {dataset_config['name']}...")
    print(f"Dataset path: {dataset_path}")
    subprocess.run([
        'python3', 'src/train.py',
        '--dataset', str(args.dataset),
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--learning_rate', str(args.learning_rate)
    ])
    
    # Run evaluation with dataset selection
    print(f"Starting evaluation on {dataset_config['name']}...")
    subprocess.run([
        'python3', 'src/evaluate.py',
        '--dataset', str(args.dataset)
    ])

if __name__ == "__main__":
    main() 