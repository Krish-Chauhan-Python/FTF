import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

def get_scheduler(name: str, optimizer: optim.Optimizer, **kwargs):
    if name == "cosine_annealing":
        return CosineAnnealingLR(optimizer, T_max=kwargs.get("T_max", 10))
    elif name == "step":
        return StepLR(optimizer, step_size=kwargs.get("step_size", 10), gamma=kwargs.get("gamma", 0.1))
    else:
        return None
