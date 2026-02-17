import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from numpy.typing import NDArray

Array2D = NDArray[np.float64]


def compute_residual_distribution(
    y_true_list: List[Array2D],
    y_pred_list: List[Array2D],
) -> Dict[str, Any]:
    all_residuals = []
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        residuals = y_true - y_pred
        all_residuals.append(residuals)

    all_residuals = np.concatenate(all_residuals, axis=0)

    residual_mean = np.mean(all_residuals, axis=0)
    residual_std = np.std(all_residuals, axis=0)

    return {
        "mean": residual_mean,
        "std": residual_std,
        "all_residuals": all_residuals,
    }


def probabilistic_multistep_forecast(
    model_forecast_fn,
    m: int,
    Y_past: Array2D,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    n_samples: int = 1,
    Z_past: Optional[Array2D] = None,
) -> Tuple[Optional[Array2D], Array2D, Optional[Array2D]]:

    if n_samples == 1:
        Zf, Yf, Xf = model_forecast_fn(m, Y_past)
        if Yf is not None:
            noise = np.random.normal(residual_mean, residual_std, size=Yf.shape)
            Yf = Yf + noise
        return Zf, Yf, Xf

    all_Yf = []
    all_Zf = []
    all_Xf = []

    for _ in range(n_samples):
        Zf, Yf, Xf = model_forecast_fn(m, Y_past)

        if Yf is not None:
            noise = np.random.normal(residual_mean, residual_std, size=Yf.shape)
            Yf = Yf + noise

        all_Yf.append(Yf)
        all_Zf.append(Zf)
        all_Xf.append(Xf)

    Yf_mean = np.mean([y for y in all_Yf if y is not None], axis=0) if all_Yf else None
    Zf_mean = np.mean([z for z in all_Zf if z is not None], axis=0) if all_Zf else None
    Xf_mean = np.mean([x for x in all_Xf if x is not None], axis=0) if all_Xf else None

    return Zf_mean, Yf_mean, Xf_mean


def autoregressive_probabilistic_forecast(
    model_predict_one_step_fn,
    m: int,
    Y_past: Array2D,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    block_samples: int = 1,
) -> Tuple[Optional[Array2D], Array2D, Optional[Array2D]]:

    Y_current = Y_past.copy()

    all_Yf = []
    all_Zf = []
    all_Xf = []

    for step in range(m):
        Zf_step, Yf_step, Xf_step = model_predict_one_step_fn(Y_current)

        if Yf_step is not None:
            last_pred = (
                Yf_step[-1:, :] if Yf_step.ndim == 2 else Yf_step[-1].reshape(1, -1)
            )

            noise = np.random.normal(residual_mean, residual_std, size=last_pred.shape)
            last_pred_noisy = last_pred + noise

            all_Yf.append(last_pred_noisy)

            Y_current = np.vstack([Y_current[1:], last_pred_noisy])

        if Zf_step is not None:
            last_z = (
                Zf_step[-1:, :] if Zf_step.ndim == 2 else Zf_step[-1].reshape(1, -1)
            )
            all_Zf.append(last_z)

        if Xf_step is not None:
            last_x = (
                Xf_step[-1:, :] if Xf_step.ndim == 2 else Xf_step[-1].reshape(1, -1)
            )
            all_Xf.append(last_x)

    Yf = np.vstack(all_Yf) if all_Yf else None
    Zf = np.vstack(all_Zf) if all_Zf else None
    Xf = np.vstack(all_Xf) if all_Xf else None

    return Zf, Yf, Xf
