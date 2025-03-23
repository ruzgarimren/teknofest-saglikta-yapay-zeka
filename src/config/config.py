from torchvision import transforms
import torch

# Data configuration
DATA_DIR = './data'
IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 4

# Model configuration
IN_CHANNELS = 3
OUT_CHANNELS = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Training configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = 'outputs'

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
]) 