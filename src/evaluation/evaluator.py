import torch
import csv
import os
import numpy as np
from tqdm import tqdm
from .metrics import compute_full_metrics

def evaluate_model(model, dataloader, device, output_csv: str, metrics_csv: str):
    model.eval()
    actuals = []
    preds = []
    
    # We do a basic evaluation loop
    with torch.no_grad():
        for sequences, targets in tqdm(dataloader, desc="Evaluating"):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(sequences)
                
            preds.extend(outputs.cpu().numpy().flatten().tolist())
            actuals.extend(targets.cpu().numpy().flatten().tolist())
            
    actual_arr = np.array(actuals, dtype=np.float64)
    pred_arr = np.array(preds, dtype=np.float64)
    
    # Save predictions
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual", "predicted", "error"])
        for a, p in zip(actual_arr, pred_arr):
            writer.writerow([a, p, p - a])
            
    # Compute metrics
    metrics = compute_full_metrics(actual_arr, pred_arr)
    
    # Save metrics
    os.makedirs(os.path.dirname(os.path.abspath(metrics_csv)), exist_ok=True)
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
        
    return metrics
