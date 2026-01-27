import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms, models

# ==========================
# 1. Dataset Class
# ==========================
class TorqueImageDataset(Dataset):
    def __init__(self, video_files, timeline_file, torque_file, transform=None, nth_frame=10):
        self.video_files = video_files
        self.transform = transform
        self.nth_frame = nth_frame

        self.timeline = self._load_timeline(timeline_file)
        self.torque_data = self._load_torque_data(torque_file)
        self.frame_map = []

        torque_timestamps = np.array([
            e["timestamp"] for e in self.torque_data
            if isinstance(e, dict) and "timestamp" in e
        ])
        if len(torque_timestamps) == 0:
            raise RuntimeError("Torque timestamps missing!")

        for video_path in self.video_files:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames_timeline = len(self.timeline)
            total_frames = min(total_frames_video, total_frames_timeline)
            for frame_idx in range(0, total_frames, self.nth_frame):
                ts = self.timeline[frame_idx]
                idx = np.abs(torque_timestamps - ts).argmin()
                target = float(self.torque_data[idx]["zeroed"][-1])
                self.frame_map.append((video_path, frame_idx, target))
            cap.release()

    def _load_timeline(self, timeline_file):
        data = np.load(timeline_file, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            obj = data.item()
            for key in ["color", "timestamps", "data", "depth"]:
                if key in obj:
                    return np.array(obj[key])
        if isinstance(data, dict):
            for key in ["color", "timestamps", "data", "depth"]:
                if key in data:
                    return np.array(data[key])
        return np.array(data)

    def _load_torque_data(self, torque_file):
        data = np.load(torque_file, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
        if isinstance(data, dict):
            key = next(iter(data))
            return data[key]
        return data

    def __len__(self):
        return len(self.frame_map)

    def __getitem__(self, idx):
        video_path, frame_idx, target = self.frame_map[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))  # fixed input size
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return frame, torch.tensor(target, dtype=torch.float32)


# ==========================
# 2. Enhanced Model – ResNet50 Regressor
# ==========================
class ResNetRegressor(nn.Module):
    def __init__(self, pretrained=True, dropout=0.4):
        super().__init__()
        print("[INFO] Loading pretrained ResNet50 backbone...")
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)


# ==========================
# 3. Transforms
# ==========================
print("[INFO] Setting up data augmentation...")
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(400, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ==========================
# 4. Dataset Collection
# ==========================
def collect_dataset_paths(root_dir, skip_ratio=0.5):
    all_tasks = sorted(glob(os.path.join(root_dir, "task_*")))
    selected_tasks = all_tasks[::int(1 / skip_ratio)] if skip_ratio < 1 else all_tasks
    dataset_entries = []
    for task_dir in selected_tasks:
        torque_path = os.path.join(task_dir, "transformed", "force_torque.npy")
        if not os.path.exists(torque_path):
            continue
        for cam_dir in sorted(glob(os.path.join(task_dir, "cam_*"))):
            video_path = os.path.join(cam_dir, "color.mp4")
            timeline_path = os.path.join(cam_dir, "timestamps.npy")
            if os.path.exists(video_path) and os.path.exists(timeline_path):
                dataset_entries.append((video_path, timeline_path, torque_path))
    return dataset_entries


# ==========================
# 5. Main Execution
# ==========================
if __name__ == "__main__":
    root_dir = r"C:\Users\Hi Krish\Downloads\RH20T_cfg1\Chemistry"
    skip_ratio = 0.1
    nth_frame = 25

    entries = collect_dataset_paths(root_dir, skip_ratio)
    datasets = [TorqueImageDataset([v], t, tq, transform=transform_train, nth_frame=nth_frame)
                for v, t, tq in entries]
    all_dataset = ConcatDataset(datasets)

    indices = np.arange(len(all_dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_data = Subset(all_dataset, train_idx)
    val_data = Subset(all_dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = ResNetRegressor(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    scaler = torch.cuda.amp.GradScaler('cuda')
    num_epochs = 15
    best_loss = float("inf")

    print("\n[INFO] Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for imgs, targets in loop:
            imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * imgs.size(0)
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        print(f"[EPOCH {epoch+1}] Train Loss: {train_loss:.6f}")

        # Validation
        model.eval()
        val_loss = 0.0
        preds, actuals = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
                preds.extend(outputs.cpu().numpy().flatten())
                actuals.extend(targets.cpu().numpy().flatten())

        val_loss /= len(val_loader.dataset)
        print(f"[EPOCH {epoch+1}] Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_resnet_regressor.pth")
            print("[INFO] Saved best model!")

    print("[INFO] Training completed. Best validation loss:", best_loss)

    # Save predictions
    df = pd.DataFrame({"actual": actuals, "predicted": preds})
    df.to_csv("validation_predictions.csv", index=False)
    print("[INFO] Predictions saved to validation_predictions.csv")
