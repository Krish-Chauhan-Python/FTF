import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms
from tqdm import tqdm


# ==========================
# 1. RGB(+pseudo-depth) Dataset
# ==========================
class TorqueRGBDDataset(Dataset):
    def __init__(self, video_files, timeline_file, torque_file, transform=None, nth_frame=5):
        """
        video_files: list of color video paths (typically length 1 here)
        timeline_file: path to timestamps.npy (or similar)
        torque_file: path to force_torque.npy
        transform: torchvision transform for a 4-channel tensor (C,H,W)
        nth_frame: sample every nth frame
        """
        self.video_files = video_files
        self.transform = transform
        self.nth_frame = nth_frame

        print(f"\n[INFO] Loading timeline from: {timeline_file}")
        self.timeline = self._load_timeline(timeline_file)

        print(f"[INFO] Loading torque data from: {torque_file}")
        self.torque_data = self._load_torque_data(torque_file)

        self.frame_map = []
        print(f"[INFO] Preparing RGB(pseudo-D) dataset for {len(video_files)} video(s)... (Sampling every {self.nth_frame}th frame)")

        for video_path in tqdm(self.video_files, desc="Videos"):
            cap_color = cv2.VideoCapture(video_path)
            if not cap_color.isOpened():
                print(f"[WARNING] Failed to open color video: {video_path}")
                continue

            total_frames_video = int(cap_color.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames_timeline = len(self.timeline) if hasattr(self.timeline, "__len__") else 0
            total_frames = min(total_frames_video, total_frames_timeline)

            print(f"[INFO] Processing video: {video_path}")
            print(f"       Frames in color video: {total_frames_video}, timestamps: {total_frames_timeline}, using: {total_frames}")

            torque_timestamps = np.array([
                e["timestamp"] for e in self.torque_data
                if isinstance(e, dict) and "timestamp" in e
            ])
            if len(torque_timestamps) == 0:
                print(f"[WARNING] No valid torque timestamps found for {video_path}")
                cap_color.release()
                continue

            for frame_idx in tqdm(range(0, total_frames, self.nth_frame),
                                  desc=f"Frames in {os.path.basename(video_path)}", leave=False):
                ts = self.timeline[frame_idx]
                idx = np.abs(torque_timestamps - ts).argmin()
                target = float(self.torque_data[idx]["zeroed"][-1])
                self.frame_map.append((video_path, frame_idx, target))

            cap_color.release()

        print(f"[INFO] RGB(pseudo-D) Dataset ready! Total frame mappings: {len(self.frame_map)}")

    def _load_timeline(self, timeline_file):
        data = np.load(timeline_file, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            obj = data.item()
            if isinstance(obj, dict):
                for key in ["color", "timestamps", "data", "depth"]:
                    if key in obj:
                        return np.array(obj[key])
            elif isinstance(obj, list):
                return np.array(obj)
        if isinstance(data, dict):
            for key in ["color", "timestamps", "data", "depth"]:
                if key in data:
                    return np.array(data[key])
        if isinstance(data, np.ndarray):
            return data
        raise TypeError(f"Unsupported timeline format in {timeline_file}")

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

        cap_color = cv2.VideoCapture(video_path)
        cap_color.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_color, color_frame = cap_color.read()
        cap_color.release()

        if not ret_color:
            raise RuntimeError(f"Failed to read color frame {frame_idx} from {video_path}")

        color_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # (H,W,3)
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0     # (H,W)
        depth_frame = gray[:, :, np.newaxis]                                               # (H,W,1)

        rgbd = np.concatenate([color_rgb, depth_frame], axis=-1)  # (H,W,4)
        assert rgbd.shape[2] == 4, f"Expected 4 channels, got {rgbd.shape}"

        rgbd_tensor = torch.from_numpy(rgbd).permute(2, 0, 1)  # (4,H,W)

        if self.transform:
            rgbd_tensor = self.transform(rgbd_tensor)

        return rgbd_tensor, torch.tensor(target, dtype=torch.float32)


# ==========================
# 2. CNN Model (4‑channel)
# ==========================
class CNNRGBDRegressor(nn.Module):
    def __init__(self,
                 in_channels=4,
                 conv_layers=8,
                 hidden_dim=1024,
                 fc_layers=2,
                 num_filters_start=32,
                 kernel_size=3,
                 output_dim=1):
        super(CNNRGBDRegressor, self).__init__()
        print(f"[INFO] Initializing CNNRGBDRegressor with {conv_layers} conv layers and {fc_layers} FC layers.")

        conv_blocks = []
        in_ch = in_channels
        out_ch = num_filters_start

        for _ in range(conv_layers):
            conv_blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1))
            conv_blocks.append(nn.BatchNorm2d(out_ch))
            conv_blocks.append(nn.ReLU(inplace=True))
            conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
            out_ch *= 2

        self.conv = nn.Sequential(*conv_blocks)
        self.flatten_dim = None

        fc_blocks = []
        for _ in range(fc_layers - 1):
            fc_blocks.append(nn.Linear(hidden_dim, hidden_dim))
            fc_blocks.append(nn.ReLU(inplace=True))
            fc_blocks.append(nn.Dropout(0.5))
        fc_blocks.append(nn.Linear(hidden_dim, output_dim))
        self.fc_layers = nn.Sequential(*fc_blocks)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        if self.flatten_dim is None:
            self.flatten_dim = x.shape[1]
            print(f"[INFO] Flattened feature size: {self.flatten_dim}")
            self._initialize_fc_layers(self.flatten_dim)
            self.fc_layers = self.fc_layers.to(x.device)
        x = self.fc_layers(x)
        return x

    def _initialize_fc_layers(self, input_dim):
        layers = []
        hidden_dim = 1024
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(0.3))
        for m in list(self.fc_layers.children())[1:]:
            layers.append(m)
        self.fc_layers = nn.Sequential(*layers)


# ==========================
# 3. Transforms (4‑channel)
# ==========================
print("[INFO] Setting up RGB(pseudo-D) transforms...")
transform = transforms.Compose([
    transforms.Resize((420, 420), antialias=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.5],
        std=[0.229, 0.224, 0.225, 0.25]
    ),
])


# ==========================
# 4. Dataset Collection
# ==========================
def collect_rgbd_dataset_paths(root_dir, skip_ratio=0.5):
    dataset_entries = []
    print(f"\n[INFO] Collecting dataset paths from root directory: {root_dir}")
    all_tasks = sorted(glob(os.path.join(root_dir, "task_*")))

    if skip_ratio < 1:
        step = max(int(1 / skip_ratio), 1)
        selected_tasks = all_tasks[::step]
    else:
        selected_tasks = all_tasks

    print(f"[INFO] Found {len(all_tasks)} total tasks. Using {len(selected_tasks)} ({100 * len(selected_tasks)/len(all_tasks):.1f}%).")

    for task_dir in tqdm(selected_tasks, desc="Task directories"):
        torque_path = os.path.join(task_dir, "transformed", "force_torque.npy")
        if not os.path.exists(torque_path):
            print(f"[WARNING] No torque file found in {task_dir}")
            continue

        camera_dirs = sorted(glob(os.path.join(task_dir, "cam_*")))
        for cam_dir in camera_dirs:
            color_path = os.path.join(cam_dir, "color.mp4")
            timeline_path = os.path.join(cam_dir, "timestamps.npy")

            if os.path.exists(color_path) and os.path.exists(timeline_path):
                dataset_entries.append((color_path, timeline_path, torque_path))
            else:
                print(f"[SKIP] Missing color video or timeline in {cam_dir}")

    print(f"[INFO] Collected {len(dataset_entries)} valid (video, timeline, torque) sets.")
    return dataset_entries


# ==========================
# 5. Main
# ==========================
if __name__ == "__main__":
    root_dir = r"C:\Users\Hi Krish\Downloads\RH20T_cfg1\Chemistry"

    skip_ratio = 0.5
    entries = collect_rgbd_dataset_paths(root_dir, skip_ratio=skip_ratio)

    if len(entries) == 0:
        raise RuntimeError("No valid entries found. Check dataset paths.")

    print(f"\n[INFO] Building dataset objects for {len(entries)} entries...")

    nth_frame = 25
    datasets = [
        TorqueRGBDDataset(
            [video_path],
            timeline_file=timeline_path,
            torque_file=torque_path,
            transform=transform,
            nth_frame=nth_frame
        )
        for (video_path, timeline_path, torque_path) in entries
    ]

    all_dataset = ConcatDataset(datasets)
    print(f"[INFO] Combined dataset size: {len(all_dataset)} frames total.")

    indices = np.arange(len(all_dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_data = Subset(all_dataset, train_idx)
    val_data = Subset(all_dataset, val_idx)
    print(f"[INFO] Train/Val split: {len(train_data)} / {len(val_data)} frames")

    print("[INFO] Creating DataLoaders...")
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    print("[INFO] DataLoaders ready!")

    # ==========================
    # 6. Train with best-model tracking
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")
    model = CNNRGBDRegressor(in_channels=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 30
    best_loss = float("inf")
    best_model_path = "best_cnn_rgbd_regressor.pth"
    best_csv_path = "best_validation_predictions.csv"

    print("\n[INFO] Starting training loop...")
    for epoch in range(num_epochs):
        # ---------- Train ----------
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for imgs, targets in loop:
            imgs = imgs.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader.dataset)
        print(f"[EPOCH {epoch+1}] Train Loss: {train_loss:.6f}")

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        preds, actuals = [], []
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False):
                imgs = imgs.to(device)
                targets = targets.to(device).unsqueeze(1)

                outputs = model(imgs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * imgs.size(0)
                preds.extend(outputs.cpu().numpy().flatten())
                actuals.extend(targets.cpu().numpy().flatten())

        val_loss /= len(val_loader.dataset)
        print(f"[EPOCH {epoch+1}] Val Loss: {val_loss:.6f}")

        # ---------- Best model & CSV ----------
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] New best model saved to {best_model_path} with Val Loss = {best_loss:.6f}")

            df = pd.DataFrame({"actual": actuals, "predicted": preds})
            df.to_csv(best_csv_path, index=False)
            print(f"[INFO] Best predictions saved to {best_csv_path}")

    print(f"[INFO] Training completed. Best validation loss: {best_loss:.6f}")
