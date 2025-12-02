from typing import List, Tuple, Any, Union, Optional, Dict
import numpy as np
from scipy import stats


def _pearson_list_2d(y_true_2d: np.ndarray, y_pred_2d: np.ndarray) -> List[float]:
    r_list: List[float] = []

    if y_true_2d is None or y_pred_2d is None:
        return r_list
    if y_true_2d.ndim != 2 or y_pred_2d.ndim != 2:
        raise ValueError(
            "Expect 2D arrays shaped (time, channels) for single-trial computation."
        )
    if (
        y_true_2d.shape[0] != y_pred_2d.shape[0]
        or y_true_2d.shape[1] != y_pred_2d.shape[1]
    ):
        raise ValueError(
            "y_true and y_pred must have the same shape for correlation computation."
        )

    for c in range(y_true_2d.shape[1]):
        t = y_true_2d[:, c]
        p = y_pred_2d[:, c]
        if np.std(t) < 1e-12 or np.std(p) < 1e-12:
            r = np.nan
        else:
            r = float(np.corrcoef(t, p)[0, 1])
        r_list.append(r)
    return r_list


def pearson_r_per_channel(
    y_true: Union[np.ndarray, List[np.ndarray]],
    y_pred: Union[np.ndarray, List[np.ndarray]],
) -> Tuple[List[Any], float]:

    trials_true: List[np.ndarray]
    trials_pred: List[np.ndarray]

    if isinstance(y_true, list) and isinstance(y_pred, list):
        trials_true = y_true
        trials_pred = y_pred
    elif isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        if y_true.ndim == 2 and y_pred.ndim == 2:
            r_list = _pearson_list_2d(y_true, y_pred)
            valid = [r for r in r_list if not (r is None or np.isnan(r))]
            r_mean = float(np.mean(valid)) if len(valid) > 0 else np.nan
            return r_list, r_mean
        elif y_true.ndim == 3 and y_pred.ndim == 3:
            trials_true = [y_true[i] for i in range(y_true.shape[0])]
            trials_pred = [y_pred[i] for i in range(y_pred.shape[0])]
        else:
            raise ValueError("Unsupported input shapes for pearson_r_per_channel.")
    else:
        raise ValueError(
            "y_true and y_pred types must match (both list or both ndarray)."
        )

    per_trial: List[List[float]] = []
    all_valid: List[float] = []
    n_trials = min(len(trials_true), len(trials_pred))
    for i in range(n_trials):
        yt = trials_true[i]
        yp = trials_pred[i]
        r_list = _pearson_list_2d(yt, yp)
        per_trial.append(r_list)
        all_valid.extend([r for r in r_list if not (r is None or np.isnan(r))])

    overall_mean = float(np.mean(all_valid)) if len(all_valid) > 0 else np.nan
    return per_trial, overall_mean


def compute_residual_statistics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    residuals = y_true - y_pred

    return {
        "residuals": residuals,
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "min": np.min(residuals),
        "max": np.max(residuals),
        "rmse": np.sqrt(np.mean(residuals**2)),
        "mae": np.mean(np.abs(residuals)),
    }


def qq_plot_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data_flat = data.flatten()
    data_clean = data_flat[np.isfinite(data_flat)]

    if len(data_clean) == 0:
        return np.array([]), np.array([])

    (theoretical_quantiles, sample_quantiles), _ = stats.probplot(
        data_clean, dist="norm"
    )

    return theoretical_quantiles, sample_quantiles


def normality_tests(data: np.ndarray) -> Dict[str, Tuple[float, float]]:
    data_flat = data.flatten()
    data_clean = data_flat[np.isfinite(data_flat)]

    results = {}

    if len(data_clean) == 0:
        results["shapiro"] = (np.nan, np.nan)
        results["ks"] = (np.nan, np.nan)
        return results

    if len(data_clean) <= 5000:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(data_clean)
            results["shapiro"] = (shapiro_stat, shapiro_p)
        except Exception:
            results["shapiro"] = (np.nan, np.nan)
    else:
        subsample = np.random.choice(data_clean, size=5000, replace=False)
        try:
            shapiro_stat, shapiro_p = stats.shapiro(subsample)
            results["shapiro"] = (shapiro_stat, shapiro_p)
        except Exception:
            results["shapiro"] = (np.nan, np.nan)

    data_standardized = (data_clean - np.mean(data_clean)) / np.std(data_clean)
    try:
        ks_stat, ks_p = stats.kstest(data_standardized, "norm")
        results["ks"] = (ks_stat, ks_p)
    except Exception:
        results["ks"] = (np.nan, np.nan)

    return results


def probability_plot_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data_flat = data.flatten()
    data_clean = data_flat[np.isfinite(data_flat)]

    if len(data_clean) == 0:
        return np.array([]), np.array([])

    data_standardized = (data_clean - np.mean(data_clean)) / np.std(data_clean)

    data_sorted = np.sort(data_standardized)

    n = len(data_sorted)
    empirical_cdf = np.arange(1, n + 1) / n

    theoretical_cdf = stats.norm.cdf(data_sorted)

    return theoretical_cdf, empirical_cdf



def compute_power_spectrum(
    signal: np.ndarray, fs: float = 1.0, nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    n_samples, n_channels = signal.shape

    if nperseg is None:
        freqs = np.fft.rfftfreq(n_samples, d=1 / fs)
        fft_result = np.fft.rfft(signal, axis=0)
        psd = np.abs(fft_result) ** 2 / n_samples
    else:
        from scipy.signal import welch

        freqs, psd = welch(signal, fs=fs, nperseg=nperseg, axis=0)

    return freqs, psd


def find_dominant_frequencies(
    freqs: np.ndarray, psd: np.ndarray, n_peaks: int = 5, min_distance: int = 5
) -> Tuple[np.ndarray, np.ndarray]:

    from scipy.signal import find_peaks

    if psd.ndim == 2:
        psd_avg = np.mean(psd, axis=1)
    else:
        psd_avg = psd

    peaks, properties = find_peaks(psd_avg, distance=min_distance)

    if len(peaks) == 0:
        return np.array([]), np.array([])

    peak_powers = psd_avg[peaks]
    sorted_indices = np.argsort(peak_powers)[::-1][:n_peaks]

    peak_freqs = freqs[peaks[sorted_indices]]
    peak_powers = peak_powers[sorted_indices]

    return peak_freqs, peak_powers


def spectral_correlation(
    freqs: np.ndarray, psd1: np.ndarray, psd2: np.ndarray
) -> float:
    if psd1.ndim == 2:
        psd1 = np.mean(psd1, axis=1)
    if psd2.ndim == 2:
        psd2 = np.mean(psd2, axis=1)
    correlation = np.corrcoef(psd1, psd2)[0, 1]
    return correlation


def autocorrelation_function(
    data: np.ndarray, max_lag: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    data_flat = data.flatten()
    data_clean = data_flat[np.isfinite(data_flat)]

    if len(data_clean) == 0:
        return np.array([]), np.array([])

    data_centered = data_clean - np.mean(data_clean)

    if max_lag is None:
        max_lag = min(len(data_centered) // 4, 40)

    acf = np.zeros(max_lag + 1)
    variance = np.var(data_centered)

    if variance == 0:
        return np.arange(max_lag + 1), acf

    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.mean(data_centered[:-lag] * data_centered[lag:]) / variance

    lags = np.arange(max_lag + 1)
    return lags, acf


def whiteness_test(data: np.ndarray, max_lag: Optional[int] = None) -> Dict[str, Any]:
    data_flat = data.flatten()
    data_clean = data_flat[np.isfinite(data_flat)]

    if len(data_clean) == 0:
        return {
            "ljung_box_stat": np.nan,
            "ljung_box_p": np.nan,
            "lags": np.array([]),
            "acf": np.array([]),
        }

    if max_lag is None:
        max_lag = min(len(data_clean) // 4, 40)

    lags, acf = autocorrelation_function(data_clean, max_lag)

    n = len(data_clean)
    lb_stat = 0.0

    for k in range(1, max_lag + 1):
        lb_stat += (acf[k] ** 2) / (n - k)

    lb_stat *= n * (n + 2)

    df = max_lag
    lb_p = 1 - stats.chi2.cdf(lb_stat, df)

    return {
        "ljung_box_stat": lb_stat,
        "ljung_box_p": lb_p,
        "lags": lags,
        "acf": acf,
    }
