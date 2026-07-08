import torch.nn as nn
from .cnn_rgb_cfg1 import CNNRGBCfg1
from .cnn_rgb_cfg2 import CNNRGBCfg2
from .cnn_depth_cfg1 import CNNDepthCfg1
from .cnn_depth_cfg2 import CNNDepthCfg2
from .efficientnet_ann import EfficientNetANN
from .resnet_ann import ResNetANN
from .yolo_pipeline import YOLOPipelineRegressor

def get_model(name: str, **kwargs) -> nn.Module:
    """Factory function to instantiate any model by name."""
    name = name.lower()
    if name == "cnn_rgb_cfg1":
        return CNNRGBCfg1(**kwargs)
    elif name == "cnn_rgb_cfg2":
        return CNNRGBCfg2(**kwargs)
    elif name == "cnn_depth_cfg1":
        return CNNDepthCfg1(**kwargs)
    elif name == "cnn_depth_cfg2":
        return CNNDepthCfg2(**kwargs)
    elif name == "efficientnet_ann":
        return EfficientNetANN(**kwargs)
    elif name == "resnet_ann":
        return ResNetANN(**kwargs)
    elif name == "yolo_pipeline":
        return YOLOPipelineRegressor(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")

__all__ = ["get_model"]
