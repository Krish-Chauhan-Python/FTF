import torch
import torch.nn as nn

class CNNRGBCfg1(nn.Module):
    """
    CNN cfg1: 6 conv blocks + 3 FC layers
    Input: RGB (3 channels), 224x224
    """
    def __init__(self):
        super().__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.3)
            )

        # 6 Convolutional Blocks
        self.features = nn.Sequential(
            conv_block(3, 32),
            nn.MaxPool2d(2, 2), # 112x112
            
            conv_block(32, 64),
            nn.MaxPool2d(2, 2), # 56x56
            
            conv_block(64, 128),
            nn.MaxPool2d(2, 2), # 28x28
            
            conv_block(128, 256),
            nn.MaxPool2d(2, 2), # 14x14
            
            conv_block(256, 512),
            nn.MaxPool2d(2, 2), # 7x7
            
            conv_block(512, 1024),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Avg Pooling to 1x1
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regression_head(x)
