import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetANN(nn.Module):
    """
    ResNet-50 (frozen) + FC regression head
    Input: RGB (3 channels), 224x224
    """
    def __init__(self):
        super().__init__()
        
        # Load ResNet backbone
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        
        # Remove FC layer
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Regression head on top of 2048-d feature vector
        self.regression_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.regression_head(features)
