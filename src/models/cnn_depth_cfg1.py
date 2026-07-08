import torch
import torch.nn as nn

class CNNDepthCfg1(nn.Module):
    """
    CNN Depth cfg1: 6 conv blocks, 4x4 pool + 2 FC layers
    Input: RGBD (4 channels), 224x224
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

        # 6 Convolutional Blocks with 4x4 pool
        self.features = nn.Sequential(
            conv_block(4, 32),
            nn.MaxPool2d(4, 4), 
            
            conv_block(32, 64),
            nn.MaxPool2d(2, 2), 
            
            conv_block(64, 128),
            nn.MaxPool2d(2, 2), 
            
            conv_block(128, 256),
            nn.MaxPool2d(2, 2), 
            
            conv_block(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            conv_block(512, 1024)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regression_head(x)
