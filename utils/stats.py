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


def fisher_z_transform(r_values: List[float]) -> Tuple[List[float], float]:
    """Apply Fisher Z transform (arctanh) to correlation values.
    Clips to [-0.9999, 0.9999] to avoid inf. Returns (z_values, z_mean)."""
    valid = [r for r in r_values if r is not None and not np.isnan(r)]
    if not valid:
        return [], float("nan")
    clipped = np.clip(valid, -0.9999, 0.9999)
    z_values = np.arctanh(clipped)
    return z_values.tolist(), float(np.mean(z_values))


def aggregate_r_per_channel(
    r_per_trial: List[Any],
    channel_names: List[str],
    apply_fisher_z: bool = False,
) -> Tuple[Dict[str, float], float]:
    """Aggregate per-trial per-channel Pearson R into per-channel means.

    Args:
        r_per_trial: Output of pearson_r_per_channel (list of lists or flat list).
        channel_names: Channel names corresponding to columns.
        apply_fisher_z: If True, apply Fisher Z transform before averaging.

    Returns:
        (per_channel_dict, overall_mean) where per_channel_dict maps
        channel name to mean (or Fisher Z mean) across trials.
    """
    n_channels = len(channel_names)
    per_channel = {}

    if not r_per_trial or n_channels == 0:
        return {}, float("nan")

    if isinstance(r_per_trial[0], list):
        # Per-trial per-channel: list of lists
        for ch_idx in range(n_channels):
            ch_vals = [
                trial[ch_idx]
                for trial in r_per_trial
                if len(trial) > ch_idx
                and trial[ch_idx] is not None
                and not np.isnan(trial[ch_idx])
            ]
            if ch_vals:
                if apply_fisher_z:
                    _, z_mean = fisher_z_transform(ch_vals)
                    per_channel[channel_names[ch_idx]] = z_mean
                else:
                    per_channel[channel_names[ch_idx]] = float(np.mean(ch_vals))
    else:
        # Single-trial: flat list
        for ch_idx, r in enumerate(r_per_trial):
            if ch_idx < n_channels and r is not None and not np.isnan(r):
                if apply_fisher_z:
                    _, z_mean = fisher_z_transform([r])
                    per_channel[channel_names[ch_idx]] = z_mean
                else:
                    per_channel[channel_names[ch_idx]] = float(r)

    all_vals = [v for v in per_channel.values() if not np.isnan(v)]
    overall_mean = float(np.mean(all_vals)) if all_vals else float("nan")
    return per_channel, overall_mean


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

    try:
        shapiro_stat, shapiro_p = stats.shapiro(data_clean)
        results["shapiro"] = (shapiro_stat, shapiro_p)
    except Exception:
        results["shapiro"] = (np.nan, np.nan)

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
    n = len(data_centered)

    if max_lag is None:
        max_lag = min(n // 4, 40)

    acf = np.zeros(max_lag + 1)
    sum_sq = np.sum(data_centered**2)

    if sum_sq < 1e-15:
        return np.arange(max_lag + 1), acf

    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            # Standard ACF estimate: divide by N, not N-lag
            # Formula: sum(x_t * x_{t-lag}) / sum(x_t^2)
            acf[lag] = np.sum(data_centered[:-lag] * data_centered[lag:]) / sum_sq

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


CLINICAL_FREQUENCY_BANDS = {
    "theta_alpha": (3, 12),
    "low_beta": (12, 16),
    "high_beta": (16, 28),
}


def bandpower(
    signal: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    nperseg: Optional[int] = None,
) -> float:
    if signal.ndim == 0:
        return 0.0

    if signal.ndim > 1:
        signal = signal.flatten()

    if nperseg is None:
        nperseg = min(len(signal), 256)

    from scipy.signal import welch

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    band_mask = (freqs >= fmin) & (freqs <= fmax)

    if not band_mask.any():
        return 0.0

    power = np.trapz(psd[band_mask], freqs[band_mask])

    return float(power)


def compare_band_power(
    signal_true: np.ndarray,
    signal_pred: np.ndarray,
    fs: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    if bands is None:
        bands = CLINICAL_FREQUENCY_BANDS

    results = {}

    for band_name, (fmin, fmax) in bands.items():
        power_true = bandpower(signal_true, fs, fmin, fmax)
        power_pred = bandpower(signal_pred, fs, fmin, fmax)

        error = np.abs(power_true - power_pred)
        ratio = power_pred / power_true if power_true > 0 else np.nan
        log_ratio = np.log10(ratio) if not np.isnan(ratio) and ratio > 0 else np.nan

        results[band_name] = {
            "true": power_true,
            "pred": power_pred,
            "error": error,
            "ratio": ratio,
            "log_ratio": log_ratio,
            "freq_range": (fmin, fmax),
        }

    return results


def compute_psd_dbs_stats(
    freqs: np.ndarray,
    psds_on: np.ndarray,
    psds_off: np.ndarray,
) -> Dict[str, Any]:
    """
    Compare DBS ON vs OFF PSD using Mann-Whitney U test.

    Each row of psds_on / psds_off is the mean PSD spectrum of one trial.
    The test compares the distribution of total-band power (integrated over
    the full frequency axis) between the two conditions.

    Returns dict with test results including U, p-value, effect size (rank-biserial r),
    and per-condition mean/std of total power (in dB).
    """
    if psds_on.ndim == 1:
        psds_on = psds_on[np.newaxis, :]
    if psds_off.ndim == 1:
        psds_off = psds_off[np.newaxis, :]

    # Integrate PSD over frequency axis per trial → scalar power per trial
    power_on = np.trapz(psds_on, freqs, axis=1)
    power_off = np.trapz(psds_off, freqs, axis=1)

    # Convert to dB scale
    power_on_db = 10 * np.log10(np.maximum(power_on, 1e-20)) + 120
    power_off_db = 10 * np.log10(np.maximum(power_off, 1e-20)) + 120

    n_on, n_off = len(power_on_db), len(power_off_db)

    result = {
        "n_on": n_on,
        "n_off": n_off,
        "mean_on": float(np.mean(power_on_db)),
        "mean_off": float(np.mean(power_off_db)),
        "std_on": float(np.std(power_on_db)),
        "std_off": float(np.std(power_off_db)),
        "delta_db": float(np.mean(power_on_db) - np.mean(power_off_db)),
    }

    if n_on >= 2 and n_off >= 2:
        U, p = stats.mannwhitneyu(power_on_db, power_off_db, alternative="two-sided")
        # Rank-biserial correlation as effect size
        r = 1 - (2 * U) / (n_on * n_off)
        result.update({"U": float(U), "p": float(p), "effect_size_r": float(r)})
    else:
        result.update({"U": np.nan, "p": np.nan, "effect_size_r": np.nan})

    return result


def compute_cross_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_lag_samples: int,
    min_lag_samples: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    y_true = (y_true - np.mean(y_true)) / (np.std(y_true) + 1e-12)
    y_pred = (y_pred - np.mean(y_pred)) / (np.std(y_pred) + 1e-12)

    lags = np.arange(min_lag_samples, max_lag_samples + 1)
    correlations = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag == 0:
            correlations[i] = np.corrcoef(y_true, y_pred)[0, 1]
        elif lag > 0 and lag < min_len:
            correlations[i] = np.corrcoef(y_true[lag:], y_pred[:-lag])[0, 1]
        else:
            correlations[i] = np.nan

    return lags, correlations


def compute_session_cross_correlation(
    true_list: List[np.ndarray],
    pred_list: List[np.ndarray],
    sampling_freq: float,
    max_lag_ms: float = 500.0,
    min_lag_ms: float = 0.0,
    channel_idx: Optional[int] = None,
) -> Dict[str, Any]:
    max_lag_samples = int(max_lag_ms * sampling_freq / 1000.0)
    min_lag_samples = int(min_lag_ms * sampling_freq / 1000.0)

    n_trials = min(len(true_list), len(pred_list))
    lags_samples = np.arange(min_lag_samples, max_lag_samples + 1)
    lags_ms = lags_samples * 1000.0 / sampling_freq

    all_correlations = np.full((n_trials, len(lags_samples)), np.nan)
    peak_correlations = np.full(n_trials, np.nan)
    optimal_lags_ms = np.full(n_trials, np.nan)

    for k in range(n_trials):
        y_true = np.asarray(true_list[k])
        y_pred = np.asarray(pred_list[k])

        if y_true is None or y_pred is None:
            continue

        if y_true.ndim == 2 and channel_idx is not None:
            y_true = y_true[:, channel_idx]
        elif y_true.ndim == 2:
            y_true = y_true.mean(axis=1)

        if y_pred.ndim == 2 and channel_idx is not None:
            y_pred = y_pred[:, channel_idx]
        elif y_pred.ndim == 2:
            y_pred = y_pred.mean(axis=1)

        _, corrs = compute_cross_correlation(
            y_true, y_pred, max_lag_samples, min_lag_samples
        )
        all_correlations[k, :] = corrs

        valid_mask = ~np.isnan(corrs)
        if valid_mask.any():
            valid_corrs = corrs[valid_mask]
            valid_lags = lags_ms[valid_mask]
            peak_idx = np.argmax(valid_corrs)
            peak_correlations[k] = valid_corrs[peak_idx]
            optimal_lags_ms[k] = valid_lags[peak_idx]

    return {
        "lags_ms": lags_ms,
        "correlations": all_correlations,
        "peak_correlations": peak_correlations,
        "optimal_lags_ms": optimal_lags_ms,
        "n_trials": n_trials,
    }


def compute_average_cross_correlation(
    session_result: Dict[str, Any],
) -> Dict[str, Any]:
    correlations = session_result["correlations"]
    lags_ms = session_result["lags_ms"]

    with np.errstate(all="ignore"):
        mean_corr = np.nanmean(correlations, axis=0)
        median_corr = np.nanmedian(correlations, axis=0)
        std_corr = np.nanstd(correlations, axis=0)

    valid_mask = ~np.isnan(mean_corr)
    if valid_mask.any():
        valid_mean = mean_corr[valid_mask]
        valid_lags = lags_ms[valid_mask]
        peak_idx = np.argmax(valid_mean)
        avg_optimal_lag = valid_lags[peak_idx]
        avg_peak_corr = valid_mean[peak_idx]
    else:
        avg_optimal_lag = np.nan
        avg_peak_corr = np.nan

    valid_lags_trial = session_result["optimal_lags_ms"]
    mask = ~np.isnan(valid_lags_trial)

    return {
        "lags_ms": lags_ms,
        "mean_correlation": mean_corr,
        "median_correlation": median_corr,
        "std_correlation": std_corr,
        "avg_optimal_lag_ms": avg_optimal_lag,
        "median_optimal_lag_ms": (
            np.nanmedian(valid_lags_trial[mask]) if mask.any() else np.nan
        ),
        "avg_peak_correlation": avg_peak_corr,
        "lag_std_ms": np.nanstd(valid_lags_trial[mask]) if mask.any() else np.nan,
    }
