import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel

class BirefNetDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BirefNetDecoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        self.conv4 = nn.Conv2d(128, num_classes, kernel_size=1)
        
        # Deep Supervision
        self.deep3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.deep2 = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = self.conv4(x3)
        
        # Upsample to input size
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=True)
        
        if self.training:
            deep3 = F.interpolate(self.deep3(x2), size=(256, 256), mode='bilinear', align_corners=True)
            deep2 = F.interpolate(self.deep2(x3), size=(256, 256), mode='bilinear', align_corners=True)
            return out, deep3, deep2
            
        return out

class SwinBirefNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(SwinBirefNet, self).__init__()
        
        # Swin Transformer Backbone
        self.swin = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
        
        if not pretrained:
            self.swin.init_weights()
            
        # Freeze early layers
        for param in list(self.swin.parameters())[:-20]:
            param.requires_grad = False
            
        # BirefNet Decoder
        self.decoder = BirefNetDecoder(in_channels=1024, num_classes=num_classes)
        
        # Auxiliary Classifier
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Resize input to 224x224 for Swin
        x_swin = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        
        # Get Swin features
        features = self.swin(x_swin).last_hidden_state
        B, L, C = features.shape
        H = W = int(L**0.5)
        features = features.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Decoder path
        if self.training:
            out, deep3, deep2 = self.decoder(features)
            aux = self.aux_classifier(features)
            return out, deep3, deep2, aux
            
        out = self.decoder(features)
        return out

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        batch_size = logits.size(0)
        
        probs = probs.reshape(batch_size, -1)
        targets = targets.reshape(batch_size, -1)
        
        intersection = (probs * targets).sum(-1)
        dice = (2. * intersection + self.smooth) / (probs.sum(-1) + targets.sum(-1) + self.smooth)
        
        return 1 - dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0]):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.weights = weights
        
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.weights[0] * bce_loss + self.weights[1] * dice_loss

def iou_score(pred, target, smooth=1e-5):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth) 