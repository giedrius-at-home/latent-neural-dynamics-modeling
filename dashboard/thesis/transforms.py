from __future__ import annotations

import numpy as np


def reshape_future_z_time_first(z: np.ndarray) -> np.ndarray:
    """
    Normalize ``Z_future`` / ``Y_future`` arrays to shape ``(n_time_steps, n_channels)``.

    Parquet rows may store either ``(steps, ch)`` or ``(ch, steps)``. The old rule only
    transposed when ``ch <= 8``, so e.g. ``(9, 80)`` was mis-read as 9 forecast steps —
    which distorts horizon curves (often looks artificially short / flat).
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("Z future must be 2D")
    r, c = z.shape
    if r < c and r <= 64 and c >= 2 * r:
        return z.T
    return z


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
    """Z-score true and all predictions on the true trial's mean/std.

    Model traces that are all-NaN (missing trial) pass through unchanged.
    """
    z_true = zscore_using_true_stats(true, true)

    def _safe_zscore(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float).ravel()
        if np.all(np.isnan(a)):
            return a
        return zscore_using_true_stats(true, arr)

    return (
        z_true,
        _safe_zscore(psid),
        _safe_zscore(dpad),
        _safe_zscore(varma),
    )


def rmse_z(z_true: np.ndarray, z_pred: np.ndarray) -> float:
    z_true = np.asarray(z_true, dtype=float).reshape(-1)
    z_pred = np.asarray(z_pred, dtype=float).reshape(-1)
    n = min(len(z_true), len(z_pred))
    if n == 0:
        return float("nan")
    d = z_true[:n] - z_pred[:n]
    return float(np.sqrt(np.mean(d**2)))
