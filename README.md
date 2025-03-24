# Teknofest Sağlıkta Yapay Zeka

This repository contains the code for the Teknofest Healthcare AI competition.

## Getting Started

To start the application, run the following command in your terminal:

```bash
chmod +x start.sh
./start.sh
```

This will:
1. Create necessary directories
2. Install required dependencies
3. Run the training and evaluation pipeline
4. Log results to Weights & Biases (wandb)

Make sure you have:
- Python 3.8 or higher installed
- Weights & Biases account (for experiment tracking)
- CUDA-capable GPU (recommended for faster training)

## Project Structure

```
.
├── data/
│   └── dataset/           # Dataset files
├── src/
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── run.py           # Main runner script
├── outputs/             # Saved models and outputs
├── requirements.txt     # Project dependencies
└── start.sh           # Start script
```

## Model Details

The project uses DeepLabV3+ with ResNet50 backbone for polyp segmentation in colonoscopy frames. The model is trained using:
- Dice Loss
- IoU Score metric
- Adam optimizer
- Cosine annealing learning rate scheduler

## Results

Training progress and results can be monitored through the Weights & Biases dashboard.