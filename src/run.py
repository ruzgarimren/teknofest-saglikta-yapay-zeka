import os
import subprocess
import wandb

def main():
    # Create necessary directories
    os.makedirs('outputs', exist_ok=True)
    
    # Run training
    print("Starting training...")
    subprocess.run(['python3', 'src/train.py'])
    
    # Run evaluation
    print("Starting evaluation...")
    subprocess.run(['python3', 'src/evaluate.py'])

if __name__ == "__main__":
    main() 