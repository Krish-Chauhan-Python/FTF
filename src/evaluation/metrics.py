import math
import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from typing import Dict, List, Tuple

def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_absolute_error(actual, predicted))

def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(actual, predicted)))

def compute_signed_error_stats(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, float]:
    err = predicted - actual
    return float(np.mean(err)), float(np.std(err))

def compute_worst_case_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    err = predicted - actual
    mu, sigma = compute_signed_error_stats(actual, predicted)
    return float(mu + 2 * sigma)

def compute_error_bin_distribution(actual: np.ndarray, predicted: np.ndarray, bins: List[float]) -> Dict[str, float]:
    abs_err = np.abs(predicted - actual)
    dist = {}
    total = len(abs_err)
    prev_b = 0.0
    for b in bins:
        count = np.sum((abs_err >= prev_b) & (abs_err < b))
        dist[f"[{prev_b}, {b})"] = float(count / total * 100)
        prev_b = b
    count = np.sum(abs_err >= prev_b)
    dist[f">={prev_b}"] = float(count / total * 100)
    return dist

def compute_safe_handling_accuracy(predicted: np.ndarray, actual: np.ndarray, object_labels: np.ndarray, percentile_range: Tuple[float, float] = (5.0, 95.0)) -> Dict[str, float]:
    # Placeholder for paper's Table VI calculation
    # Requires object-specific historical torque data to compute bounds. 
    # Here we simulate the interface.
    unique_objects = np.unique(object_labels)
    accs = {}
    for obj in unique_objects:
        mask = (object_labels == obj)
        p = predicted[mask]
        a = actual[mask]
        if len(a) == 0: continue
        
        # In a real scenario, lower_bound and upper_bound are computed from the historical distribution
        # For evaluation, we assume predictions within +/- sigma of actual are safe
        std = np.std(a) if len(a) > 1 else 0.1
        safe_mask = (p >= a - std) & (p <= a + std)
        accs[str(obj)] = float(np.mean(safe_mask) * 100.0)
    
    if len(accs) > 0:
        accs["average"] = float(np.mean(list(accs.values())))
    return accs

def compute_full_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    if len(actual) == 0:
        return {}
        
    err = predicted - actual
    abs_err = np.abs(err)
    nonzero = np.abs(actual) > 1e-12
    mape = float(np.mean(abs_err[nonzero] / np.abs(actual[nonzero])) * 100.0) if np.any(nonzero) else float("nan")

    mse = float(mean_squared_error(actual, predicted))
    rmse = float(math.sqrt(mse))

    return {
        "count": float(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": mse,
        "rmse": rmse,
        "median_ae": float(median_absolute_error(actual, predicted)),
        "mape_percent": mape,
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
        "explained_variance": float(explained_variance_score(actual, predicted)) if len(actual) > 1 else float("nan"),
        "max_error": float(max_error(actual, predicted)) if len(actual) > 1 else float(abs_err.max() if len(abs_err) else float("nan")),
        "bias_mean_error": float(np.mean(err)),
        "pearson_r": float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 1 else float("nan"),
    }
