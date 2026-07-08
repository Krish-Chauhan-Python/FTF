import sys
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import find_task_dirs, collect_video_entries, RH20TTorqueDataset
from src.data.depth_dataset import RH20TDepthTorqueDataset
from src.data.yolo_dataset import YOLOCroppedTorqueDataset
from src.data.transforms import get_val_transforms
from src.models import get_model
from src.evaluation.evaluator import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--test-root", type=str, default="./data/test", help="Test data root")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    task_dirs = find_task_dirs(train_root=args.test_root)
    entries = collect_video_entries(task_dirs)
    
    img_size = config["data"].get("image_size", 224)
    val_tf = get_val_transforms(input_size=img_size)
    
    if "depth" in args.model:
        dataset = RH20TDepthTorqueDataset(entries, transform=val_tf, image_size=img_size)
    elif "yolo" in args.model:
        yolo_path = config["yolo"].get("weights", "weights/best.pt")
        dataset = YOLOCroppedTorqueDataset(entries, yolo_model_path=yolo_path, transform=val_tf, image_size=img_size)
    else:
        dataset = RH20TTorqueDataset(entries, transform=val_tf, image_size=img_size)
        
    dataloader = DataLoader(dataset, batch_size=config["training"].get("batch_size", 256), shuffle=False)
    
    output_csv = f"outputs/{args.model}_predictions.csv"
    metrics_csv = f"outputs/{args.model}_metrics.csv"
    
    print(f"[INFO] Evaluating {args.model}...")
    metrics = evaluate_model(model, dataloader, device, output_csv, metrics_csv)
    
    print("=" * 50)
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
