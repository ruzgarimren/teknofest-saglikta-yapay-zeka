import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as album
import wandb
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils

def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map

def reverse_one_hot(image):
    x = np.argmax(image, axis = -1)
    return x

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x

def crop_image(image, true_dimensions):
    return album.CenterCrop(p=1, height=true_dimensions[0], width=true_dimensions[1])(image=image)

class EndoscopyDataset(torch.utils.data.Dataset):
    def __init__(self, df, class_rgb_values=None, augmentation=None, preprocessing=None):
        self.image_paths = df['png_image_path'].tolist()
        self.mask_paths = df['png_mask_path'].tolist()
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.image_paths)

def get_validation_augmentation():
    test_transform = [
        album.PadIfNeeded(min_height=288, min_width=384, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)

def main():
    wandb.init(project="polyp-segmentation", job_type="evaluation")

    DATA_DIR = 'data/dataset'
    metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
    metadata_df = metadata_df[['frame_id', 'png_image_path', 'png_mask_path']]
    metadata_df['png_image_path'] = metadata_df['png_image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata_df['png_mask_path'] = metadata_df['png_mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

    valid_df = metadata_df.sample(frac=0.1, random_state=42)

    class_dict = pd.read_csv(os.path.join(DATA_DIR, 'class_dict.csv'))
    class_names = class_dict['class_names'].tolist()
    class_rgb_values = class_dict[['r','g','b']].values.tolist()

    select_classes = ['background', 'polyp']
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    model = torch.load('outputs/best_model.pth')
    preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet50", "imagenet")

    test_dataset = EndoscopyDataset(
        valid_df,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    test_dataset_vis = EndoscopyDataset(
        valid_df,
        class_rgb_values=select_class_rgb_values,
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    model.eval()

    test_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=smp.utils.losses.DiceLoss(),
        metrics=[smp.utils.metrics.IoU(threshold=0.5)],
        device=DEVICE,
        verbose=True,
    )

    valid_logs = test_epoch.run(test_dataloader)
    print("Evaluation on Test Data: ")
    print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
    print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

    wandb.log({
        "test/iou_score": valid_logs['iou_score'],
        "test/dice_loss": valid_logs['dice_loss']
    })

    for idx in range(len(test_dataset)):
        image, gt_mask = test_dataset[idx]
        image_vis = test_dataset_vis[idx][0].astype('uint8')
        true_dimensions = image_vis.shape
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        
        with torch.no_grad():
            pred_mask = model(x_tensor)
            pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        
        pred_mask = np.transpose(pred_mask,(1,2,0))
        pred_polyp_heatmap = crop_image(pred_mask[:,:,select_classes.index('polyp')], true_dimensions)['image']
        pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values), true_dimensions)['image']
        
        gt_mask = np.transpose(gt_mask,(1,2,0))
        gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values), true_dimensions)['image']

        wandb.log({
            "predictions": wandb.Image(
                np.hstack([image_vis, gt_mask, pred_mask]),
                caption=f"Sample {idx}: Original | Ground Truth | Prediction"
            ),
            "heatmap": wandb.Image(
                pred_polyp_heatmap,
                caption=f"Sample {idx}: Polyp Heatmap"
            )
        })

    wandb.finish()

if __name__ == "__main__":
    main() 