import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetANN(nn.Module):
    """
    EfficientNet-B0 (frozen) + ANN regressor
    Input: RGB (3 channels), 224x224
    """
    def __init__(self):
        super().__init__()
        
        # Load EfficientNet backbone
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.regression_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.backbone(x) # 1280-d
        return self.regression_head(features)
