import os
import argparse
import torch
import wandb
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from data.dataset import MultiDataset, get_preprocessing
from config.config import MODEL_CONFIG, DATASET_CONFIGS, get_data_config

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--dataset', type=int, choices=[1, 2, 3], required=True,
                      help='Dataset ID to evaluate on (1, 2, or 3)')
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_config = DATASET_CONFIGS[args.dataset]
    data_config = get_data_config(args.dataset)
    
    wandb.init(project="polyp-segmentation", job_type="evaluation")
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(MODEL_CONFIG['ENCODER'], MODEL_CONFIG['ENCODER_WEIGHTS'])
    
    test_dataset = MultiDataset(
        dataset_id=args.dataset,
        data_dir=data_config['data_dir'],
        train=False,
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    
    test_dataset_vis = MultiDataset(
        dataset_id=args.dataset,
        data_dir=data_config['data_dir'],
        train=False
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    model_path = os.path.join('outputs', f'best_model_dataset_{args.dataset}.pth')
    if not os.path.exists(model_path):
        raise ValueError(f"No model found for dataset {args.dataset}")
    
    model = smp.DeepLabV3Plus(
        encoder_name=MODEL_CONFIG['ENCODER'],
        encoder_weights=MODEL_CONFIG['ENCODER_WEIGHTS'],
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(model_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    test_iou = 0.0
    test_loss = 0.0
    
    with torch.no_grad():
        for images, masks in test_dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            pred = torch.sigmoid(outputs)
            pred = (pred > 0.5).float()
            
            intersection = (pred * masks).sum()
            union = pred.sum() + masks.sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            test_iou += iou.item()
            
            loss = smp.utils.losses.DiceLoss()(outputs, masks)
            test_loss += loss.item()
            
            # Log predictions
            wandb.log({
                "predictions": wandb.Image(
                    np.hstack([
                        images[0].cpu().numpy().transpose(1, 2, 0),
                        masks[0].cpu().numpy().transpose(1, 2, 0),
                        pred[0].cpu().numpy().transpose(1, 2, 0)
                    ]),
                    caption="Original | Ground Truth | Prediction"
                )
            })
    
    test_iou /= len(test_dataloader)
    test_loss /= len(test_dataloader)
    
    print(f"Evaluation on {dataset_config['name']}:")
    print(f"Mean IoU Score: {test_iou:.4f}")
    print(f"Mean Dice Loss: {test_loss:.4f}")
    
    wandb.log({
        "test/iou_score": test_iou,
        "test/dice_loss": test_loss
    })
    
    wandb.finish()

if __name__ == "__main__":
    main() 