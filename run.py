import os
import argparse
from src.main import main

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation model on multiple datasets')
    parser.add_argument('--dataset', type=int, choices=[1, 2, 3], required=True,
                      help='Dataset ID to train on (1, 2, or 3)')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.00008,
                      help='Learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args) 