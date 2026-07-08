import sys
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split

# Add root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import find_task_dirs, collect_video_entries, RH20TTorqueDataset
from src.data.depth_dataset import RH20TDepthTorqueDataset
from src.data.yolo_dataset import YOLOCroppedTorqueDataset
from src.data.transforms import get_train_transforms, get_val_transforms, get_depth_train_transforms
from src.models import get_model
from src.training.trainer import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. cnn_rgb_cfg1, yolo_pipeline, etc.)")
    parser.add_argument("--data-root", type=str, default="./data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--yolo-weights", type=str, default="weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    config["training"]["epochs"] = args.epochs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model Setup
    model = get_model(args.model).to(device)
    
    if args.dry_run:
        print("[INFO] Performing dry run...")
        batch_size = config["training"].get("batch_size", 256)
        # Fake data shape based on model
        channels = 4 if "depth" in args.model else 3
        img_size = config["data"].get("image_size", 224)
        x = torch.randn(2, channels, img_size, img_size).to(device)
        y = model(x)
        print(f"[INFO] Smoke test successful! Output shape: {y.shape}")
        return

    # Data Setup
    task_dirs = find_task_dirs(train_root=args.data_root)
    entries = collect_video_entries(task_dirs)
    
    img_size = config["data"].get("image_size", 224)
    train_tf = get_train_transforms(input_size=img_size)
    val_tf = get_val_transforms(input_size=img_size)
    
    # Choose dataset type
    if "depth" in args.model:
        # We need a depth transform if it doesn't match standard RGB
        # For simplicity, we use the raw dataset and let it resize internally
        dataset = RH20TDepthTorqueDataset(entries, transform=val_tf, image_size=img_size)
    elif "yolo" in args.model:
        dataset = YOLOCroppedTorqueDataset(entries, yolo_model_path=args.yolo_weights, transform=val_tf, image_size=img_size)
    else:
        dataset = RH20TTorqueDataset(entries, transform=val_tf, image_size=img_size)
        
    val_split = config["training"].get("val_split", 0.2)
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    
    train_set, val_set = random_split(
        dataset, 
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(config["training"].get("seed", 42))
    )
    
    # In a real scenario, you'd apply different transforms for train vs val.
    # To do that properly, you might wrap the subset in a class that applies the transform.
    
    batch_size = config["training"].get("batch_size", 256)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    save_path = f"weights/{args.model}_best.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"[INFO] Training {args.model} for {args.epochs} epochs...")
    train_model(model, train_loader, val_loader, device, config, save_path)

if __name__ == "__main__":
    main()
