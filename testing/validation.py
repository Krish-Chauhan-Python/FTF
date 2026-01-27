import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 5 CSVs (EffNet is the first one)
csv_paths = [
    r'C:\Users\Hi Krish\Desktop\Coding\Python\FTF\Efficientnet based\validation_predictions.csv',
    r'C:\Users\Hi Krish\Desktop\Coding\Python\FTF\in_channels=3,conv_layers=6,hidden_dim=512,fc_layers=3,num_filters_start=32,kernel_size=2,output_dim=1)\validation_results.csv',
    r'C:\Users\Hi Krish\Desktop\Coding\Python\FTF\in_channels=3,conv_layers=8,hidden_dim=1024,fc_layers=2,num_filters_start=32,kernel_size=2,output_dim=1\validation_results.csv',
    r'C:\Users\Hi Krish\Desktop\Coding\Python\FTF\resnetbased 3 epoc\validation_predictions_optimized.csv',
    r'C:\Users\Hi Krish\Desktop\Coding\Python\FTF\resnetbased 25 epocs\validation_predictions.csv'
]

model_names = [
    "EffNet base",
    "CNN cfg1",
    "CNN cfg2",
    "ResNet 3e",
    "ResNet 25e"
]

# -------- helper functions (from single‑file code, generalized) --------
def iqr_mask(pct_err, k=1.0):
    """Return boolean mask keeping points within IQR band."""
    q1 = np.percentile(pct_err, 25)
    q3 = np.percentile(pct_err, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (pct_err >= lower) & (pct_err <= upper)

def smooth_series(arr, win=35):
    arr = np.asarray(arr, dtype=float)
    if len(arr) > win:
        return uniform_filter1d(arr, size=win, mode="nearest")
    return arr

# -------- load all models, apply same formulas --------
all_abs_curves = []
all_rmse_curves = []
all_pct_curves = []
metrics = []  # (rmse, mae, mean_pct_error, mean_abs_pct_error)

for path in csv_paths:
    try:
        df = pd.read_csv(path)

        actual = df["actual"].to_numpy(dtype=float)
        # allow either "predicted" or "prediction_0"
        if "predicted" in df.columns:
            pred = df["predicted"].to_numpy(dtype=float)
        else:
            pred_cols = [c for c in df.columns if c.startswith("prediction_")]
            if not pred_cols:
                raise ValueError(f"No predicted column in {path}")
            pred = df[pred_cols[-1]].to_numpy(dtype=float)

        err = pred - actual
        abs_err = np.abs(err)
        pct_err = np.abs(err / np.where(actual != 0, actual, 1) * 100.0)

        # IQR‑based outlier removal on percentage error (same as your script)
        mask = iqr_mask(pct_err, k=1.0)

        actual_f   = actual[mask]
        pred_f     = pred[mask]
        err_f      = err[mask]
        abs_err_f  = abs_err[mask]
        pct_err_f  = pct_err[mask]

        # global metrics on filtered data
        rmse = np.sqrt(mean_squared_error(actual_f, pred_f))
        mae  = mean_absolute_error(actual_f, pred_f)
        mean_pct_error     = np.mean(pct_err_f)
        mean_abs_pct_error = np.mean(np.abs(pct_err_f))
        metrics.append((rmse, mae, mean_pct_error, mean_abs_pct_error))

        # smoothed curves for plotting
        abs_curve  = smooth_series(abs_err_f, win=35)
        sq_curve   = smooth_series(err_f ** 2, win=35)
        rmse_curve = np.sqrt(sq_curve)
        pct_curve  = smooth_series(pct_err_f, win=35)

        all_abs_curves.append(abs_curve)
        all_rmse_curves.append(rmse_curve)
        all_pct_curves.append(pct_curve)

        print(f"{os.path.basename(path)} -> kept {mask.sum()}/{len(mask)} after IQR")

    except Exception as e:
        print(f"Error loading {path}: {e}")
        all_abs_curves.append(None)
        all_rmse_curves.append(None)
        all_pct_curves.append(None)
        metrics.append((None, None, None, None))

# align lengths
valid_lengths = [len(c) for c in all_abs_curves if c is not None]
common_len = min(valid_lengths)
x = np.arange(common_len)

colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
eff_name = "EffNet base"

def is_effnet(name):
    return name == eff_name

# ---------- Figure 1: absolute error (EffNet highlighted) ----------
plt.figure(figsize=(12, 6))
for i, (name, curve, m) in enumerate(zip(model_names, all_abs_curves, metrics)):
    if curve is None:
        continue
    rmse, mae, mean_pct, mean_abs_pct = m
    y = curve[:common_len]
    label = f"{name} (RMSE={rmse:.4f}, MAE={mae:.4f}, %MAE={mean_abs_pct:.2f}%)"

    if is_effnet(name):
        plt.plot(x, y, color="tab:red", linewidth=2.5,
                 label=label + "  <-- EffNet", zorder=3)
    else:
        plt.plot(x, y, color=colors[i], linewidth=1.2, alpha=0.35,
                 label=label, zorder=1)

plt.xlabel("Sample index (filtered, common range)")
plt.ylabel("Absolute error |pred - actual|")
plt.title("Smoothed Absolute Error Across Models (IQR outliers removed)")
plt.grid(alpha=0.25)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# ---------- Figure 2: per-sample RMSE ----------
plt.figure(figsize=(12, 6))
for i, (name, curve) in enumerate(zip(model_names, all_rmse_curves)):
    if curve is None:
        continue
    y = curve[:common_len]
    if is_effnet(name):
        plt.plot(x, y, color="tab:red", linewidth=2.5,
                 label=name + " (EffNet)", zorder=3)
    else:
        plt.plot(x, y, color=colors[i], linewidth=1.2, alpha=0.35,
                 label=name, zorder=1)

plt.xlabel("Sample index (filtered, common range)")
plt.ylabel("Per-sample RMSE")
plt.title("Smoothed RMSE Across Models (IQR outliers removed, EffNet highlighted)")
plt.grid(alpha=0.25)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# ---------- Figure 3: percentage MAE (|% error|) ----------
plt.figure(figsize=(12, 6))
for i, (name, curve) in enumerate(zip(model_names, all_pct_curves)):
    if curve is None:
        continue
    y = curve[:common_len]
    if is_effnet(name):
        plt.plot(x, y, color="tab:red", linewidth=2.5,
                 label=name + " (EffNet)", zorder=3)
    else:
        plt.plot(x, y, color=colors[i], linewidth=1.2, alpha=0.35,
                 label=name, zorder=1)

plt.xlabel("Sample index (filtered, common range)")
plt.ylabel("Absolute percentage error (%)")
plt.title("Smoothed %MAE Across Models (IQR outliers removed, EffNet highlighted)")
plt.grid(alpha=0.25)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()
