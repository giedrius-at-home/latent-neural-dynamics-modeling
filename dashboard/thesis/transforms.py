from __future__ import annotations

import numpy as np


def zscore_using_true_stats(true: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Map `x` into z-scored units defined by the ground-truth trial `true`:
    z(x) = (x - mean(true)) / std(true).
    """
    true = np.asarray(true, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    mu = float(np.mean(true))
    sigma = float(np.std(true))
    if sigma < 1e-12:
        sigma = 1.0
    return (x - mu) / sigma


def z_true_and_preds(
    true: np.ndarray,
    psid: np.ndarray,
    dpad: np.ndarray,
    varma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score true and all predictions on the true trial's mean/std."""
    z_true = zscore_using_true_stats(true, true)
    return (
        z_true,
        zscore_using_true_stats(true, psid),
        zscore_using_true_stats(true, dpad),
        zscore_using_true_stats(true, varma),
    )


def rmse_z(z_true: np.ndarray, z_pred: np.ndarray) -> float:
    z_true = np.asarray(z_true, dtype=float).reshape(-1)
    z_pred = np.asarray(z_pred, dtype=float).reshape(-1)
    n = min(len(z_true), len(z_pred))
    if n == 0:
        return float("nan")
    d = z_true[:n] - z_pred[:n]
    return float(np.sqrt(np.mean(d**2)))
