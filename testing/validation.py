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
pct_err = err / np.where(actual != 0, actual, 1) * 100.0

# --- Outlier removal on percentage error ---
# IQR method: keep values within [Q1 - k*IQR, Q3 + k*IQR]
k = 1.5  # adjust: smaller = more aggressive trimming
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
rmse = mean_squared_error(actual_f, pred_f)
mae = mean_absolute_error(actual_f, pred_f)
mean_pct_error = np.mean(pct_err_f)
mean_abs_pct_error = np.mean(np.abs(pct_err_f))

print(f"Filtered RMSE: {rmse:.4f}, MAE: {mae:.4f}")
print(f"Filtered mean % error: {mean_pct_error:.2f}%")
print(f"Filtered mean absolute % error: {mean_abs_pct_error:.2f}%")

x = np.arange(len(pct_err_f))

# 1) Percentage error figure
plt.figure(figsize=(10, 4))
plt.plot(x, pct_err_f, label="Percentage Error (%)")
plt.xlabel("Sample index (filtered)")
plt.ylabel("% Error")
plt.title(
    f"Percentage Error for {pred_col}\n"
    f"Mean abs % error (filtered) = {mean_abs_pct_error:.2f}%"
)
plt.grid(True)
plt.ylim(-100, 100)  # optional visualization limit
plt.tight_layout()
plt.show()

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
