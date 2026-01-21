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
from torchvision import transforms, models
from tqdm import tqdm

# ==========================
# 1. Dataset Class
# ==========================
class TorqueImageDataset(Dataset):
    def __init__(self, video_files, timeline_file, torque_file, transform=None, nth_frame=5):
        self.video_files = video_files
        self.transform = transform
        self.nth_frame = nth_frame

        print(f"\n[INFO] Loading timeline from: {timeline_file}")
        self.timeline = self._load_timeline(timeline_file)

        print(f"[INFO] Loading torque data from: {torque_file}")
        self.torque_data = self._load_torque_data(torque_file)

        self.frame_map = []
        print(f"[INFO] Preparing dataset for {len(video_files)} video(s)... (Sampling every {self.nth_frame}th frame)")

        for video_path in tqdm(self.video_files, desc="Videos"):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[WARNING] Failed to open video: {video_path}")
                continue

            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames_timeline = len(self.timeline) if hasattr(self.timeline, "__len__") else 0
            total_frames = min(total_frames_video, total_frames_timeline)
            print(f"[INFO] Processing video: {video_path}")
            print(f"       Frames in video: {total_frames_video}, timestamps: {total_frames_timeline}, using: {total_frames}")

            torque_timestamps = np.array([
                e["timestamp"] for e in self.torque_data
                if isinstance(e, dict) and "timestamp" in e
            ])
            if len(torque_timestamps) == 0:
                print(f"[WARNING] No valid torque timestamps found for {video_path}")
                continue

            for frame_idx in tqdm(range(0, total_frames, self.nth_frame),
                                  desc=f"Frames in {os.path.basename(video_path)}", leave=False):
                ts = self.timeline[frame_idx]
                idx = np.abs(torque_timestamps - ts).argmin()
                target = float(self.torque_data[idx]["zeroed"][-1])
                self.frame_map.append((video_path, frame_idx, target))

            cap.release()

        print(f"[INFO] Dataset ready! Total frame mappings: {len(self.frame_map)}")

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
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows() 
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0  # HWC → CHW and normalize to [0,1]
        if self.transform:
            frame_tensor = self.transform(frame_tensor)
        return frame_tensor, torch.tensor(target, dtype=torch.float32)


# ==========================
# 2. CNN Model
# ==========================
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNRegressor(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 conv_layers=6, 
                 hidden_dim=512, 
                 fc_layers=3,
                 num_filters_start=32, 
                 kernel_size=2, 
                 output_dim=1):
        """
        Parameters:
            in_channels (int): Number of input channels (3 for RGB images)
            conv_layers (int): Number of convolutional layers
            hidden_dim (int): Number of units in hidden FC layers
            fc_layers (int): Number of fully connected layers after flattening
            num_filters_start (int): Filters in first conv layer (doubles each layer)
            kernel_size (int): Size of convolution kernel
            output_dim (int): Output dimension (1 for regression)
        """
        super(CNNRegressor, self).__init__()
        print(f"[INFO] Initializing CNNRegressor with {conv_layers} conv layers and {fc_layers} FC layers.")

        # --------------------------
        # 1. Convolutional layers
        # --------------------------
        conv_blocks = []
        in_ch = in_channels
        out_ch = num_filters_start

        for i in range(conv_layers):
            conv_blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1))
            conv_blocks.append(nn.BatchNorm2d(out_ch))
            conv_blocks.append(nn.ReLU(inplace=True))
            conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
            out_ch *= 2  # double filters at each stage

        self.conv = nn.Sequential(*conv_blocks)

        # --------------------------
        # 2. Fully connected layers
        # --------------------------
        # We'll infer the flattened size dynamically during forward()
        self.flatten_dim = None  

        fc_blocks = []
        for i in range(fc_layers - 1):
            fc_blocks.append(nn.Linear(hidden_dim, hidden_dim))
            fc_blocks.append(nn.ReLU(inplace=True))
            fc_blocks.append(nn.Dropout(0.3))
        fc_blocks.append(nn.Linear(hidden_dim, output_dim))
        self.fc_layers = nn.Sequential(*fc_blocks)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)

        if self.flatten_dim is None:
            self.flatten_dim = x.shape[1]
            print(f"[INFO] Flattened feature size: {self.flatten_dim}")
            # Initialize FC layers dynamically
            self._initialize_fc_layers(self.flatten_dim)
            # Move to same device as input
            self.fc_layers = self.fc_layers.to(x.device)

        x = self.fc_layers(x)
        return x

    def _initialize_fc_layers(self, input_dim):
        """Rebuild the FC layers once we know input size."""
        layers = []
        hidden_dim = next(m.out_features for m in self.fc_layers if isinstance(m, nn.Linear))
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(0.3))
        for m in list(self.fc_layers.children())[1:]:
            layers.append(m)
        self.fc_layers = nn.Sequential(*layers)

# ==========================
# 3. Data Preprocessing
# ==========================
print("[INFO] Setting up image transforms...")
transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((420,420), antialias=True),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


# ==========================
# 4. Dataset Collection
# ==========================
def collect_dataset_paths(root_dir, skip_ratio=0.5):
    dataset_entries = []
    print(f"\n[INFO] Collecting dataset paths from root directory: {root_dir}")
    all_tasks = sorted(glob(os.path.join(root_dir, "task_*")))

    # Skip a fraction of task folders
    selected_tasks = all_tasks[::int(1 / skip_ratio)] if skip_ratio < 1 else all_tasks
    print(f"[INFO] Found {len(all_tasks)} total tasks. Using {len(selected_tasks)} ({100 * len(selected_tasks)/len(all_tasks):.1f}%).")

    for task_dir in tqdm(selected_tasks, desc="Task directories"):
        torque_path = os.path.join(task_dir, "transformed", "force_torque.npy")
        if not os.path.exists(torque_path):
            print(f"[WARNING] No torque file found in {task_dir}")
            continue

        camera_dirs = sorted(glob(os.path.join(task_dir, "cam_*")))
        for cam_dir in camera_dirs:
            video_path = os.path.join(cam_dir, "color.mp4")
            timeline_path = os.path.join(cam_dir, "timestamps.npy")
            if os.path.exists(video_path) and os.path.exists(timeline_path):
                dataset_entries.append((video_path, timeline_path, torque_path))
            else:
                print(f"[SKIP] Missing video or timeline in {cam_dir}")

    print(f"[INFO] Collected {len(dataset_entries)} valid (video, timeline, torque) sets.")
    return dataset_entries


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    root_dir = os.getenv('dataset_path')

    # Skip half the folders
    skip_ratio = 0.5
    entries = collect_dataset_paths(root_dir, skip_ratio=skip_ratio)

    if len(entries) == 0:
        raise RuntimeError("No valid (video, timeline, torque) entries found. Check dataset paths.")

    print(f"\n[INFO] Building dataset objects for {len(entries)} entries...")

    # Change nth_frame here as needed
    nth_frame = 25
    datasets = [TorqueImageDataset([v], t, tq, transform=transform, nth_frame=nth_frame)
                for v, t, tq in entries]

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
    # 6. Train Model
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Using device: {device}")
    model = CNNRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 15
    print("\n[INFO] Starting training loop...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for imgs, targets in loop:
            imgs, targets = imgs.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            loop.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[EPOCH {epoch+1}] Training Loss: {epoch_loss:.6f}")

    torch.save(model.state_dict(), "cnn_regressor_weights.pth")
    print("[INFO] Model weights saved as cnn_regressor_weights.pth")

    # ==========================
    # 7. Validation & Save Predictions
    # ==========================
    print("\n[INFO] Starting validation...")
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc="Evaluating"):
            imgs = imgs.to(device)
            outputs = model(imgs).cpu().numpy().flatten()
            preds.extend(outputs)
            actuals.extend(targets.numpy().flatten())

    print("[INFO] Validation completed. Saving results...")

    results_file = "validation_results.csv"
    run_name = f"pred_run_{len(glob('cnn_regressor_weights*.pth'))}"

    df = pd.DataFrame({'actual': actuals})
    if os.path.exists(results_file):
        existing = pd.read_csv(results_file)
        if len(existing) == len(df):
            existing[f'prediction_{len(existing.columns)}'] = preds
            existing.to_csv(results_file, index=False)
            print(f"[INFO] Appended new predictions column to {results_file}")
        else:
            print("[INFO] Validation size changed — creating new results file.")
            df['prediction_0'] = preds
            df.to_csv(results_file, index=False)
    else:
        df['prediction_0'] = preds
        df.to_csv(results_file, index=False)
        print(f"[INFO] Created new {results_file}")

    print("\n[INFO] All done!")
