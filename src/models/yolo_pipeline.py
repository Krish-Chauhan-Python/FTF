import torch
import torch.nn as nn
from torchvision import models

class YOLOPipelineRegressor(nn.Module):
    """
    Stage 2 of the YOLO pipeline: FC regression head on cropped object.
    Takes in a 224x224 crop (after YOLO stage 1) and outputs torque.
    Uses ResNet backbone (default ResNet18) for feature extraction.
    """
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        in_features = self.backbone.fc.in_features
        
        # Replace original fc with regression head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.backbone(x)
