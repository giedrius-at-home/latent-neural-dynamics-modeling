"""Shared helpers for sec8 (per-trial x per-feature test-set heatmaps).

Pure data/loader/metric helpers extracted from the notebook so the notebook
cells stay thin (load -> compute -> plot). `results_root` and the figure
output dir are passed in explicitly rather than closed over.
"""

import numpy as np
import matplotlib.pyplot as plt

from modules.style import panel_label
from modules.loaders import EXP_Z_AS_BEHAVIOR, resolve_input_channels
from modules.lib.loaders import load_split_results_required
from modules.lib.transforms import reshape_future_z_time_first
from modules.sec2_common import load_channel_names

# Behavioural output features (fixed) — x-tick labels for the behavioural heatmap.
BEHAV_CHANNELS = ["tracing_velocity_x", "tracing_acceleration_magnitude"]


# ---------------------------------------------------------------------------
# Loaders / channel resolution
# ---------------------------------------------------------------------------


def load_test_results(results_root, session):
    """Load test-set results for one session. Returns the results dict."""
    return load_split_results_required(
        results_root, session["variant"], session["run_ts"], "test"
    )


def channels_for(session, res):
    """Input (Y) channel names with YAML fallback — current PSID parquets have
    empty ``input_channels`` and need the training YAML to name the 60 ECoG bands."""
    return resolve_input_channels(res, session["variant"])


def neural_y_features(results_root, session):
    """Y (ECoG) feature names in model order, read from the PSID train parquet."""
    return load_channel_names(
        results_root, session["session"], EXP_Z_AS_BEHAVIOR, "Y_features"
    )


def get_channel_names(results_root, session):
    """Neural channel names sorted by electrode then band."""
    channels = neural_y_features(results_root, session)

    # Sort: electrode number, then band frequency
    def sort_key(ch):
        parts = (
            ch.replace("ECOG_", "").replace("_raw", "").replace("_env", "").split("_", 1)
        )
        electrode = int(parts[0])
        band = parts[1] if len(parts) > 1 else ""
        # Extract start frequency for ordering
        freq = 0
        for token in band.split("_"):
            try:
                freq = int(token)
                break
            except ValueError:
                continue
        return (electrode, freq, band)

    return sorted(channels, key=sort_key)


def trial_label(res, i):
    """B11.T3 ON etc."""
    stim = res["stim"][i].upper() if res["stim"][i] else "?"
    return f'B{res["block"][i]}.T{res["trial"][i]} {stim}'


def sorted_trial_indices(res):
    """Return trial indices sorted by stim (OFF first), then block, then trial."""
    n = len(res["stim"])
    indices = list(range(n))

    def key(i):
        stim_order = 0 if res["stim"][i] == "off" else 1
        return (stim_order, res["block"][i], res["trial"][i])

    return sorted(indices, key=key)


# ---------------------------------------------------------------------------
# Per-trial metric computers
# ---------------------------------------------------------------------------


def compute_per_trial_rmse_neural(res, channel_order, raw_channels):
    """Compute RMSE for each trial × neural channel from Y vs Yp.

    Returns (n_trials, n_channels) array in the order of channel_order.
    """
    channels = raw_channels
    # Build channel index mapping: channel_order[j] -> position in Y columns
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}
    col_indices = [ch_to_idx[ch] for ch in channel_order]

    n_trials = len(res["Y"])
    n_ch = len(channel_order)
    rmse_matrix = np.full((n_trials, n_ch), np.nan)

    for i in range(n_trials):
        y_true = np.array(res["Y"][i])  # (T, n_channels)
        y_pred = np.array(res["Yp"][i])  # (T, n_channels)
        for j, ci in enumerate(col_indices):
            diff = y_true[:, ci] - y_pred[:, ci]
            rmse_matrix[i, j] = np.sqrt(np.mean(diff**2))

    return rmse_matrix


def compute_per_trial_pearson_neural(res, channel_order, raw_channels):
    """Extract per-trial Pearson R for neural channels from precomputed results.

    Returns (n_trials, n_channels) array in the order of channel_order.
    """
    channels = raw_channels
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}
    col_indices = [ch_to_idx[ch] for ch in channel_order]

    n_trials = len(res["pearson_per_channel"])
    n_ch = len(channel_order)
    r_matrix = np.full((n_trials, n_ch), np.nan)

    for i in range(n_trials):
        ppc = res["pearson_per_channel"][i]
        if ppc is not None:
            ppc = np.array(ppc)
            for j, ci in enumerate(col_indices):
                if ci < len(ppc):
                    r_matrix[i, j] = ppc[ci]

    return r_matrix


def compute_per_trial_metrics_behavioral(res):
    """Compute per-trial Pearson R and RMSE for behavioral channels (Z vs Zp).

    Returns (pearson_matrix, rmse_matrix) each of shape (n_trials, 2).
    """
    n_trials = len(res["Z"])
    n_ch = 2  # velocity_x, accel_mag
    r_matrix = np.full((n_trials, n_ch), np.nan)
    rmse_matrix = np.full((n_trials, n_ch), np.nan)

    for i in range(n_trials):
        z_true = np.array(res["Z"][i])  # (T, 2)
        z_pred = np.array(res["Zp"][i])  # (T, 2)
        for j in range(min(n_ch, z_true.shape[1])):
            zt = z_true[:, j]
            zp = z_pred[:, j]
            # RMSE
            rmse_matrix[i, j] = np.sqrt(np.mean((zt - zp) ** 2))
            # Pearson R
            mask = np.isfinite(zt) & np.isfinite(zp)
            if mask.sum() > 2:
                r_matrix[i, j] = np.corrcoef(zt[mask], zp[mask])[0, 1]

    return r_matrix, rmse_matrix


def compute_per_trial_metrics_neural_forecast(res, channel_order, raw_channels):
    """Per-trial × per-channel Pearson R and RMSE for the neural forecast.

    Uses Y_future_true / Y_future_pred (shape (T_forecast, n_neural)). Channels are
    reordered to match channel_order. Returns (r_matrix, rmse_matrix), both
    (n_trials, len(channel_order)), or (None, None) if forecast data absent.
    """
    if "Y_future_true" not in res or res.get("Y_future_true") is None:
        return None, None

    channels = raw_channels
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}
    col_indices = [ch_to_idx[ch] for ch in channel_order]

    yft = res["Y_future_true"]
    yfp = res["Y_future_pred"]
    n_trials = len(yft)
    n_ch = len(channel_order)
    r_matrix = np.full((n_trials, n_ch), np.nan)
    rmse_matrix = np.full((n_trials, n_ch), np.nan)

    for i in range(n_trials):
        if yft[i] is None or yfp[i] is None:
            continue
        try:
            T = reshape_future_z_time_first(np.asarray(yft[i], dtype=float))
            P = reshape_future_z_time_first(np.asarray(yfp[i], dtype=float))
        except ValueError:
            continue
        if T.shape != P.shape:
            continue
        for j, ci in enumerate(col_indices):
            if ci >= T.shape[1]:
                continue
            t_vec = T[:, ci]
            p_vec = P[:, ci]
            mask = np.isfinite(t_vec) & np.isfinite(p_vec)
            if mask.sum() < 3:
                continue
            mu = float(np.mean(t_vec[mask]))
            sigma = float(np.std(t_vec[mask]))
            if sigma < 1e-12:
                sigma = 1.0
            zt = (t_vec - mu) / sigma
            zp = (p_vec - mu) / sigma
            rmse_matrix[i, j] = float(np.sqrt(np.nanmean((zt - zp) ** 2)))
            r_matrix[i, j] = float(np.corrcoef(t_vec[mask], p_vec[mask])[0, 1])

    return r_matrix, rmse_matrix


def compute_per_trial_metrics_forecast(res):
    """Compute per-trial Pearson R and RMSE for behavioral forecasts.

    Uses Z_future_true and Z_future_pred from the results.
    Returns (pearson_matrix, rmse_matrix) each of shape (n_trials, 2), or None if no forecast data.
    """
    if "Z_future_true" not in res or res.get("Z_future_true") is None:
        return None, None

    zft = res["Z_future_true"]
    zfp = res["Z_future_pred"]
    n_trials = len(zft)
    n_ch = 2
    r_matrix = np.full((n_trials, n_ch), np.nan)
    rmse_matrix = np.full((n_trials, n_ch), np.nan)

    for i in range(n_trials):
        if zft[i] is None or zfp[i] is None:
            continue
        zt = np.array(zft[i])  # (T_forecast, 2)
        zp = np.array(zfp[i])
        if zt.ndim < 2 or zp.ndim < 2:
            continue
        for j in range(min(n_ch, zt.shape[1])):
            z_t = zt[:, j]
            z_p = zp[:, j]
            rmse_matrix[i, j] = np.sqrt(np.mean((z_t - z_p) ** 2))
            mask = np.isfinite(z_t) & np.isfinite(z_p)
            if mask.sum() > 2:
                r_matrix[i, j] = np.corrcoef(z_t[mask], z_p[mask])[0, 1]

    return r_matrix, rmse_matrix


# ---------------------------------------------------------------------------
# Figure builder + representative-trial picker
# ---------------------------------------------------------------------------


def behavioral_pair_heatmap(
    out_dir, sess_label, t_labels, r_sorted, rmse_sorted, png_name, *, r_title, rmse_title
):
    """Two-panel behavioural heatmap: Pearson r (left) + RMSE (right)."""
    n_trials = len(t_labels)
    fig_h = max(3.5, n_trials * 0.15 + 1.2)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, fig_h), sharey=True)

    # Panel A: Pearson r
    ax = axes[0]
    im_r = ax.imshow(
        r_sorted,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(BEHAV_CHANNELS)))
    ax.set_xticklabels(BEHAV_CHANNELS, rotation=20, ha="right", fontsize=7)
    ax.set_yticks(np.arange(n_trials))
    ax.set_yticklabels(t_labels, fontsize=6)
    ax.set_ylabel("trial (block.trial stim)")
    fig.colorbar(im_r, ax=ax, shrink=0.7, label="Pearson's correlation coefficient")
    panel_label(ax, "A", r_title)

    # Panel B: RMSE
    ax = axes[1]
    rmse_vmax = float(np.nanmax(rmse_sorted)) if np.isfinite(rmse_sorted).any() else 1.0
    im_e = ax.imshow(
        rmse_sorted,
        cmap="Blues",
        vmin=0,
        vmax=rmse_vmax,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(BEHAV_CHANNELS)))
    ax.set_xticklabels(BEHAV_CHANNELS, rotation=20, ha="right", fontsize=7)
    fig.colorbar(im_e, ax=ax, shrink=0.7, label="Normalized RMSE")
    panel_label(ax, "B", rmse_title)

    fig.savefig(out_dir / png_name)
    plt.show()


def top_trials_for_sec2(sess_label: str, sd: dict, k: int = 3) -> None:
    res = sd["res"]
    n_trials = len(res["stim"])
    stim = [res["stim"][i] for i in range(n_trials)]
    blocks = [res["block"][i] for i in range(n_trials)]
    trials = [res["trial"][i] for i in range(n_trials)]

    has_y_f = sd.get("r_yfore") is not None
    has_z_f = sd.get("r_zfore") is not None
    mean_r_y = (
        np.nanmean(sd["r_yfore"], axis=1) if has_y_f else np.full(n_trials, np.nan)
    )
    mean_r_z = (
        np.nanmean(sd["r_zfore"], axis=1) if has_z_f else np.full(n_trials, np.nan)
    )
    mean_rmse_y = (
        np.nanmean(sd["rmse_yfore"], axis=1) if has_y_f else np.full(n_trials, np.nan)
    )
    mean_rmse_z = (
        np.nanmean(sd["rmse_zfore"], axis=1) if has_z_f else np.full(n_trials, np.nan)
    )

    print(f"  -- {sess_label} --")
    for cond in ("off", "on"):
        idxs = [i for i in range(n_trials) if stim[i] == cond]
        if not idxs:
            continue
        # Rank by (mean neural forecast r + mean behavioural forecast r) — picks trials
        # where both modalities look good in the forecast.
        scored = []
        for i in idxs:
            score = 0.0
            if has_y_f and np.isfinite(mean_r_y[i]):
                score += mean_r_y[i]
            if has_z_f and np.isfinite(mean_r_z[i]):
                score += mean_r_z[i]
            scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        top = scored[:k]
        for rank, (_, i) in enumerate(top, 1):
            print(
                f"    DBS-{cond.upper()} #{rank}: B{blocks[i]}.T{trials[i]} "
                f"(neural r={mean_r_y[i]:.3f}, behav r={mean_r_z[i]:.3f}, "
                f"neural RMSE={mean_rmse_y[i]:.3f}, behav RMSE={mean_rmse_z[i]:.3f})"
            )
