import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

results_file = "validation_results.csv"
if not os.path.exists(results_file):
    raise FileNotFoundError(f"{results_file} not found. Run the training script first.")

df = pd.read_csv(results_file)
print(df)

actual = df["actual"].to_numpy()
pred_cols = [c for c in df.columns if c.startswith("prediction_")]
if not pred_cols:
    raise ValueError("No prediction_ columns found in validation_results.csv")

pred_col = pred_cols[-1]
pred = df[pred_col].to_numpy()

err = pred - actual
abs_err = np.abs(err)
pct_err = np.abs(err / np.where(actual != 0, actual, 1) * 100.0)

# --- Outlier removal on percentage error ---
# IQR method: keep values within [Q1 - k*IQR, Q3 + k*IQR]
k = 1  # adjust: smaller = more aggressive trimming
q1 = np.percentile(pct_err, 25)
q3 = np.percentile(pct_err, 75)
iqr = q3 - q1
lower = q1 - k * iqr
upper = q3 + k * iqr

mask = (pct_err >= lower) & (pct_err <= upper)

print(f"Original samples: {len(pct_err)}, kept after outlier removal: {mask.sum()}")

# filtered arrays
actual_f = actual[mask]
pred_f   = pred[mask]
err_f    = err[mask]
abs_err_f = np.abs(err_f)
pct_err_f = pct_err[mask]

# recompute metrics on filtered data
rmse = np.sqrt(mean_squared_error(actual_f, pred_f))  # Fixed: use sqrt for RMSE
mae = mean_absolute_error(actual_f, pred_f)
mean_pct_error = np.mean(pct_err_f)
mean_abs_pct_error = np.mean(np.abs(pct_err_f))

print(f"Filtered RMSE: {rmse:.4f}, MAE: {mae:.4f}")


x = np.arange(len(pct_err_f))



# 2) Absolute error figure
plt.figure(figsize=(10, 4))
plt.plot(x, abs_err_f, label="Absolute Error")
plt.xlabel("Sample index (filtered)")
plt.ylabel("Absolute Error")
plt.title(f"Absolute Error for {pred_col} (outliers removed)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3) Global RMSE / MAE figure
plt.figure(figsize=(10, 4))
plt.axhline(rmse, color="red", linestyle="--", label=f"RMSE = {rmse:.3f}")
plt.axhline(mae, color="green", linestyle="--", label=f"MAE = {mae:.3f}")
plt.xlabel("Sample index (filtered)")
plt.ylabel("Error")
plt.title(f"Global RMSE / MAE for {pred_col} (outliers removed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4) NEW: High Error Indicator (>200% error)
plt.figure(figsize=(12, 4))
high_error_mask = pct_err_f > 200  # True if % error > 200%
plt.plot(x, high_error_mask.astype(int), marker='o', markersize=4, linewidth=2, 
         label="High Error", color='red', alpha=0.8)
plt.xlabel("Sample index (filtered)")
plt.ylabel("High Error Flag")
plt.title(f"High Error Detectionfor {pred_col}\n"
          f"High error samples: {high_error_mask.sum()}/{len(high_error_mask)} ({100*high_error_mask.mean():.1f}%)")
plt.grid(True, alpha=0.3)
plt.yticks([0, 1], ['False', 'True'])
plt.legend()
plt.tight_layout()
plt.show()

# Print summary
print(f"\nHigh error (>200%) summary for {pred_col}:")
print(f"  Total high error samples: {high_error_mask.sum()}")
print(f"  Percentage: {100*high_error_mask.mean():.1f}%")
print(f"  High error indices (first 10): {np.where(high_error_mask)[0][:10]}")
