import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml

from .scheduler import get_scheduler

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, grad_accum_steps):
    model.train()
    running_loss = 0.0
    total = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)
            
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        loss_to_backprop = loss / max(1, grad_accum_steps)
        scaler.scale(loss_to_backprop).backward()
        
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
        
    return running_loss / max(1, total)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            if len(targets.shape) == 1:
                targets = targets.unsqueeze(1)
                
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            
    return running_loss / max(1, total)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict,
    save_path: str
):
    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 25)
    lr = float(train_cfg.get("learning_rate", 1e-4))
    
    if train_cfg.get("optimizer", "adam") == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    # Scheduler
    scheduler = None
    if "resnet_extended" in config and config["resnet_extended"].get("scheduler") == "cosine_annealing":
        t_max = config["resnet_extended"].get("scheduler_t_max", 10)
        scheduler = get_scheduler("cosine_annealing", optimizer, T_max=t_max)
        
    best_val_loss = float("inf")
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    
    for epoch in range(1, epochs + 1):
        print(f"\n[EPOCH {epoch}/{epochs}]")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_accum_steps)
        val_loss = validate(model, val_loader, criterion, device)
        
        if scheduler is not None:
            scheduler.step()
            
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Saved best model with loss {best_val_loss:.6f} to {save_path}")
            
    return best_val_loss
