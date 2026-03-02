import streamlit as st
import numpy as np
import pickle
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any, List
import re
from training.components.tester import Tester
from utils.classification import load_precomputed_results
from utils.polars import get_scalar_value, convert_series_to_list
import plotly.graph_objects as go

from dashboard.backbone import (
    PALETTE,
    PLOT_STYLE,
    PLOT_COLOR,
    create_base_time_series_figure,
    create_base_psd_line_figure,
    add_margin_visualization,
    render_styled_table,
)
from utils.stats import (
    compute_power_spectrum,
    compute_residual_statistics,
    normality_tests,
    qq_plot_data,
)

BASELINE_COLOR = "#00E5FF"


def variant_short_name(variant: str) -> str:
    """Extract model type label from variant name, e.g. 'psid_PDI1_S2' -> 'PSID'."""
    return str(variant).split("_")[0].upper()


def find_baseline_variants(current_variant: str, all_variants: List[str]) -> List[str]:
    """Find baseline (VARMA) variants matching the same subject/session."""
    subject_match = re.search(r"(PDI\d+)_(?:S)?(\d+)", current_variant)
    baseline_search_str = (
        f"varma_{subject_match.group(1)}_S{subject_match.group(2)}"
        if subject_match
        else "varma"
    )
    return [
        v for v in all_variants if baseline_search_str in v and v != current_variant
    ]


def get_project_root(cfg_path: Path) -> Path:
    """Walk up from cfg_path to find the project root (parent of 'results/')."""
    for p in cfg_path.parents:
        if (p / "results").exists():
            return p
    return cfg_path


def find_config_path(project_root: Path, variant_name: str) -> Optional[Path]:
    """Recursively search for a variant's YAML config file under training/setups/ and classification/setups/."""
    for setup_dir in ["training/setups", "classification/setups"]:
        base = project_root / setup_dir
        if base.exists():
            matches = list(base.rglob(f"{variant_name}.yaml"))
            if matches:
                return matches[0]
    return None


def list_variants(results_root: Path) -> List[str]:
    if not results_root.exists():
        return []
    return sorted([p.name for p in results_root.iterdir() if p.is_dir()])


def list_run_timestamps(variant_dir: Path) -> List[str]:

    ts = set()
    for p in variant_dir.glob("val_results_*"):
        name = p.name
        if name.startswith("val_results_"):
            ts.add(name.replace("val_results_", ""))
    for p in variant_dir.glob("model_*.pkl"):
        name = p.name
        if name.startswith("model_") and name.endswith(".pkl"):
            ts.add(name.replace("model_", "").replace(".pkl", ""))
    for p in variant_dir.glob("model_*_metadata.json"):
        name = p.name
        if name.startswith("model_") and name.endswith("_metadata.json"):
            ts.add(name.replace("model_", "").replace("_metadata.json", ""))
    ts_pattern = re.compile(r"^\d{8}_\d{6}$")
    for p in variant_dir.iterdir():
        if p.is_dir() and ts_pattern.match(p.name):
            ts.add(p.name)
    return sorted(list(ts))


def config_for_variant(project_root: Path, variant_name: str) -> Optional[Path]:
    training_setups = project_root / "training" / "setups"
    for cfg in training_setups.rglob(f"{variant_name}.yaml"):
        return cfg

    classification_setups = project_root / "classification" / "setups"
    for cfg in classification_setups.rglob(f"{variant_name}.yaml"):
        return cfg

    return None


def check_precomputed_results(variant_dir: Path, run_ts: str) -> Dict[str, bool]:
    available = {}
    run_dir = variant_dir / run_ts

    for split in ["train", "val", "test"]:
        pickle_path = run_dir / f"{split}_results.pkl"
        legacy_parquet_path = variant_dir / f"{split}_results_{run_ts}"
        new_parquet_path = variant_dir / split / f"test_results_{run_ts}.parquet"

        available[split] = (
            pickle_path.exists()
            or legacy_parquet_path.exists()
            or new_parquet_path.exists()
        )

    return available


def compute_predictions_selective(
    config_path: str, run_timestamp: str, splits_to_compute: tuple
):
    tester = Tester.from_config_file(config_path, run_timestamp=run_timestamp)
    tester.run_predictions_selective(list(splits_to_compute))

    config_path_obj = Path(config_path)
    variant = config_path_obj.stem

    project_root = config_path_obj.parent.parent.parent
    results_dir = project_root / "results" / variant / run_timestamp

    for split_name in splits_to_compute:
        if split_name in tester.results:
            save_split_results(results_dir, split_name, tester.results[split_name])
            print(
                f"Saved {split_name} results to {results_dir / f'{split_name}_results.pkl'}"
            )

    return tester.results


def save_split_results(results_dir: Path, split_name: str, split_results: dict):
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{split_name}_results.pkl"

    with open(results_file, "wb") as f:
        pickle.dump(split_results, f, protocol=pickle.HIGHEST_PROTOCOL)


def compute_forecast_for_trial(
    config_path: str,
    run_ts: str,
    Y_trial: np.ndarray,
    Z_trial: Optional[np.ndarray],
    chunk_margin: Optional[float],
) -> Dict[str, Any]:
    tester = Tester.from_config_file(config_path, run_timestamp=run_ts)
    tester._load_model_for_run()
    f_res = tester.framework.model.validate_forecast(
        [Y_trial],
        Z_list=[Z_trial] if Z_trial is not None else None,
        margin=chunk_margin,
    )

    return {
        "m": f_res.get("m", 0),
        "margin_samples": f_res.get("margin_samples", 0),
        "Y_future_true": f_res.get("Y_future_true", [None])[0],
        "Y_future_pred": f_res.get("Y_future_pred", [None])[0],
        "Y_concat_for_plot": f_res.get("Y_concat_for_plot", [None])[0],
        "Z_future_true": f_res.get("Z_future_true", [None])[0],
        "Z_future_pred": f_res.get("Z_future_pred", [None])[0],
        "Z_concat_for_plot": f_res.get("Z_concat_for_plot", [None])[0],
        "pearson_per_channel": f_res.get("pearson_per_channel", [None])[0],
        "pearson_per_channel_Z": f_res.get("pearson_per_channel_Z", [None])[0],
    }


def get_trial_time_axis(
    split_res: Dict[str, Any], trial_idx: int, n_samples: int, t_offset: float = 0.0
) -> np.ndarray:
    meta_time = split_res.get("time", [])
    t_abs = meta_time[trial_idx] if meta_time and len(meta_time) > trial_idx else None

    if t_abs is None or (hasattr(t_abs, "__len__") and len(t_abs) != n_samples):
        md_list = split_res.get("margined_duration", [])
        dur = md_list[trial_idx] if md_list else None

        if dur is not None:
            t_abs = np.linspace(0.0, float(dur), n_samples)
        else:
            t_abs = np.arange(n_samples)
    else:
        t_abs = np.array(t_abs)

    return t_abs + t_offset


def transpose_if_needed(data: np.ndarray, expected_len: int) -> np.ndarray:
    if (
        data.ndim == 2
        and data.shape[0] != expected_len
        and data.shape[1] == expected_len
    ):
        return data.T
    return data


def rescale_to_reference(
    pred: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    pred = np.asarray(pred).flatten()
    ref = np.asarray(ref).flatten()

    pred_mean = np.mean(pred)
    pred_std = np.std(pred)
    ref_mean = np.mean(ref)
    ref_std = np.std(ref)

    if pred_std < 1e-10:
        return np.full_like(pred, ref_mean)

    pred_rescaled = (pred - pred_mean) / pred_std * ref_std + ref_mean
    return pred_rescaled


def render_prediction_psd_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sampling_rate: int = 60,
    channel_name: str = "",
    baseline_preds: Optional[np.ndarray] = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
):
    """Generic Power Spectral Density analysis for neural or behavioral signals."""
    if y_true is None or y_pred is None:
        return

    freqs_t, psd_t = compute_power_spectrum(y_true, sampling_rate)
    freqs_p, psd_p = compute_power_spectrum(y_pred, sampling_rate)

    psd_t_val = psd_t.squeeze() if psd_t.ndim > 1 else psd_t
    psd_p_val = psd_p.squeeze() if psd_p.ndim > 1 else psd_p

    # Convert to dB while avoiding log(0)
    psd_t_db = 10 * np.log10(psd_t_val + 1e-20) + 120
    psd_p_db = 10 * np.log10(psd_p_val + 1e-20) + 120

    fig = create_base_psd_line_figure(
        x_label="Frequency (Hz)",
        y_label="Power (dB)",
    )

    fig.add_trace(
        go.Scatter(
            x=freqs_t,
            y=psd_t_db,
            name="True PSD",
            mode="lines",
            line=dict(color=PLOT_COLOR.stim_off, width=PLOT_STYLE.line_width_normal),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=freqs_p,
            y=psd_p_db,
            name=f"{model_name} PSD",
            mode="lines",
            line=dict(
                color=PLOT_COLOR.stim_on, width=PLOT_STYLE.line_width_normal, dash="dot"
            ),
        )
    )

    if baseline_preds is not None:
        freqs_b, psd_b = compute_power_spectrum(baseline_preds, sampling_rate)
        psd_b_val = psd_b.squeeze() if psd_b.ndim > 1 else psd_b
        psd_b_db = 10 * np.log10(psd_b_val + 1e-20) + 120
        fig.add_trace(
            go.Scatter(
                x=freqs_b,
                y=psd_b_db,
                name=f"{baseline_name} PSD",
                mode="lines",
                line=dict(
                    color=BASELINE_COLOR,
                    width=PLOT_STYLE.line_width_normal,
                    dash="dash",
                ),
            )
        )

    st.plotly_chart(fig, use_container_width=True)


def render_residual_plot(
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    t_abs: np.ndarray,
    channel_name: str,
    unit: str = "",
    chunk_margin: float = 0.0,
    baseline_preds: Optional[np.ndarray] = None,
    baseline_name: str = "Baseline",
    rescale: bool = False,
):
    """Generic residual plot for neural or behavioral signals."""
    if y_true_c is None or y_pred_c is None:
        return

    if rescale:
        y_pred_c = rescale_to_reference(y_pred_c, y_true_c)
        if baseline_preds is not None:
            baseline_preds = rescale_to_reference(baseline_preds, y_true_c)

    residuals = y_true_c - y_pred_c
    unit_label = f" ({unit})" if unit else ""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_abs,
            y=residuals,
            mode="lines",
            name="Model Residuals",
            line=dict(color=PALETTE.strawberry_red, width=1.2),
        )
    )
    rmse = np.sqrt(np.mean(residuals**2))

    baseline_caption = ""
    if baseline_preds is not None:
        baseline_residuals = y_true_c - baseline_preds
        fig.add_trace(
            go.Scatter(
                x=t_abs,
                y=baseline_residuals,
                mode="lines",
                name=f"{baseline_name} Residuals",
                line=dict(color=BASELINE_COLOR, width=1.2, dash="dash"),
                opacity=0.7,
            )
        )
        base_rmse = np.sqrt(np.mean(baseline_residuals**2))
        baseline_caption = f" | {baseline_name} RMSE={base_rmse:.3f}{unit_label}"

    if chunk_margin > 0:
        add_margin_visualization(fig, t_abs, chunk_margin)

    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        margin=dict(l=60, r=80, t=40, b=60),
        xaxis_title="Time (s)",
        yaxis_title=f"Residual{unit_label}",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Residuals (Prediction Errors) — {channel_name} (RMSE={rmse:.3f}{unit_label}){baseline_caption}"
    )


def render_statistics_table(
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    r_ch: float,
    channel_name: str,
    unit: str = "",
    baseline_preds: Optional[np.ndarray] = None,
    baseline_r: Optional[float] = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
    rescale: bool = False,
):
    """Unified statistics table for neural or behavioral signals."""
    if y_true_c is None or y_pred_c is None:
        return

    if rescale:
        y_pred_c = rescale_to_reference(y_pred_c, y_true_c)
        if baseline_preds is not None:
            baseline_preds = rescale_to_reference(baseline_preds, y_true_c)

    residuals = y_true_c - y_pred_c
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true_c - np.mean(y_true_c)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    unit_label = f" ({unit})" if unit else ""

    st.markdown(f"### Statistics — {channel_name}")

    def render_metrics(label, r, r2, rmse_val, mae_val):
        st.markdown(f"**{label}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pearson r", f"{r:.4f}" if not np.isnan(r) else "N/A")
        c2.metric("R²", f"{r2:.4f}")
        c3.metric(f"RMSE{unit_label}", f"{rmse_val:.3f}")
        c4.metric(f"MAE{unit_label}", f"{mae_val:.3f}")

    render_metrics(model_name, r_ch, r_squared, rmse, mae)

    if baseline_preds is not None:
        base_residuals = y_true_c - baseline_preds
        b_mae = np.mean(np.abs(base_residuals))
        b_rmse = np.sqrt(np.mean(base_residuals**2))
        b_ss_res = np.sum(base_residuals**2)
        b_r2 = 1 - (b_ss_res / ss_tot) if ss_tot != 0 else 0
        b_r = baseline_r if baseline_r is not None else np.nan
        render_metrics(baseline_name, b_r, b_r2, b_rmse, b_mae)


def _show_residual_stats_row(stats: dict, label: str = ""):
    """Helper for residual diagnostics table."""
    if label:
        st.markdown(f"**{label}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{stats['mean']:.4f}")
    c2.metric("Std Dev", f"{stats['std']:.4f}")
    c3.metric("Min", f"{stats['min']:.4f}")
    c4.metric("Max", f"{stats['max']:.4f}")


def render_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    channel_name: str,
    baseline_preds: Optional[np.ndarray] = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
    rescale: bool = False,
):
    """Generic residual diagnostics (normality, whiteness)."""
    if y_true is None or y_pred is None:
        return

    if rescale:
        y_pred = rescale_to_reference(y_pred, y_true)
        if baseline_preds is not None:
            baseline_preds = rescale_to_reference(baseline_preds, y_true)

    res_stats = compute_residual_statistics(y_true, y_pred)
    residuals = res_stats["residuals"]

    st.markdown(f"### Residual Diagnostics — {channel_name}")

    if baseline_preds is not None:
        base_res_stats = compute_residual_statistics(y_true, baseline_preds)
        base_residuals = base_res_stats["residuals"]
        _show_residual_stats_row(res_stats, model_name)
        _show_residual_stats_row(base_res_stats, baseline_name)
    else:
        _show_residual_stats_row(res_stats)

    st.markdown("#### Normality Test")
    col1, col2 = st.columns(2)
    with col1:
        shapiro_stat, shapiro_p = normality_tests(residuals)["shapiro"]
        st.metric(
            f"Shapiro-Wilk ({model_name})",
            f"p = {shapiro_p:.4f}",
            delta=f"stat = {shapiro_stat:.4f}",
        )
    if baseline_preds is not None:
        with col2:
            b_shapiro_stat, b_shapiro_p = normality_tests(base_residuals)["shapiro"]
            st.metric(
                f"Shapiro-Wilk ({baseline_name})",
                f"p = {b_shapiro_p:.4f}",
                delta=f"stat = {b_shapiro_stat:.4f}",
            )

    st.markdown("#### Q-Q Plot")
    residuals_std = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)
    tq, sq = qq_plot_data(residuals_std)
    fig_qq = go.Figure()
    fig_qq.add_trace(
        go.Scatter(
            x=tq,
            y=sq,
            mode="markers",
            name=model_name,
            marker=dict(size=4, color=PALETTE.twilight_indigo, opacity=0.6),
        )
    )
    if baseline_preds is not None:
        b_resids_std = (base_residuals - np.mean(base_residuals)) / (
            np.std(base_residuals) + 1e-12
        )
        btq, bsq = qq_plot_data(b_resids_std)
        fig_qq.add_trace(
            go.Scatter(
                x=btq,
                y=bsq,
                mode="markers",
                name=baseline_name,
                marker=dict(size=4, color=BASELINE_COLOR, opacity=0.4),
            )
        )

    min_q = min(np.min(tq), np.min(sq))
    max_q = max(np.max(tq), np.max(sq))
    fig_qq.add_trace(
        go.Scatter(
            x=[min_q, max_q],
            y=[min_q, max_q],
            mode="lines",
            name="Normal",
            line=dict(color="black", dash="dash"),
        )
    )
    fig_qq.update_layout(
        template="plotly_white",
        margin=dict(l=60, r=80, t=40, b=60),
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
    )
    st.plotly_chart(fig_qq, use_container_width=True)


def select_baseline(
    cfg_path: Path,
    key_prefix: str,
    split_name: str,
    trial_idx: int,
):
    project_root = get_project_root(cfg_path)
    results_root = project_root / "results"
    all_variants = list_variants(results_root)
    current_variant = cfg_path.stem

    baseline_variants = find_baseline_variants(current_variant, all_variants)
    model_label = variant_short_name(current_variant) if cfg_path else "Model"

    baseline_res = None
    baseline_name = "Baseline"
    selected_variant = None

    if baseline_variants:
        st.markdown("#### Baseline Comparison")
        selected = st.selectbox(
            "Select Baseline Model",
            options=["None"] + baseline_variants,
            index=1 if baseline_variants else 0,
            key=f"{key_prefix}_baseline_select",
        )

        if selected != "None":
            baseline_dir = results_root / selected
            timestamps = list_run_timestamps(baseline_dir)

            if timestamps:
                baseline_ts = st.selectbox(
                    "Baseline run timestamp",
                    options=timestamps,
                    index=len(timestamps) - 1,
                    key=f"{key_prefix}_baseline_ts",
                )

                baseline_res = load_precomputed_results(
                    baseline_dir, baseline_ts, split_name
                )
                baseline_name = variant_short_name(selected)
                selected_variant = selected

    return baseline_res, baseline_name, model_label, selected_variant


def get_channel(
    data: np.ndarray,
    channel_idx: int,
    t_abs: np.ndarray,
) -> np.ndarray:
    """Extract a single channel from a multi-channel array, transposing if needed."""
    data = transpose_if_needed(data, len(t_abs))
    n_chan = data.shape[1] if data.ndim == 2 else 1
    return data.squeeze() if n_chan == 1 else data[:, channel_idx]


def get_baseline_channel(
    baseline_res: Optional[Dict],
    key: str,
    trial_idx: int,
    channel_idx: int,
    t_abs: np.ndarray,
    true_c: np.ndarray,
) -> tuple:
    """Extract baseline predictions for one channel and compute correlation.

    Returns (baseline_pred_c, baseline_r). Both are None/np.nan if unavailable.
    """
    if baseline_res is None:
        return None, None

    preds_list = baseline_res.get(key, [])
    if not preds_list or trial_idx >= len(preds_list) or preds_list[trial_idx] is None:
        return None, None

    preds = np.array(preds_list[trial_idx])
    preds = transpose_if_needed(preds, len(t_abs))
    n_chan = preds.shape[1] if preds.ndim == 2 else 1

    if channel_idx >= n_chan:
        return None, None

    pred_c = preds.squeeze() if n_chan == 1 else preds[:, channel_idx]

    try:
        r = np.corrcoef(true_c.flatten(), pred_c.flatten())[0, 1]
    except Exception:
        r = np.nan

    return pred_c, r


def render_analysis(
    true_c: np.ndarray,
    pred_c: np.ndarray,
    t_abs: np.ndarray,
    channel_name: str,
    r_ch: float,
    sampling_rate: int = 60,
    unit: str = "",
    chunk_margin: float = 0.0,
    baseline_pred_c: Optional[np.ndarray] = None,
    baseline_r: Optional[float] = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
    rescale: bool = False,
    show_psd: bool = True,
    diagnostics_label: str = "Residual Diagnostics & Normality Tests",
):
    """Full analysis pipeline: PSD → residual plot → statistics → diagnostics expander."""
    if true_c is None or pred_c is None:
        return

    if show_psd:
        st.markdown("#### PSD Analysis: True vs Predicted")
        render_prediction_psd_analysis(
            true_c,
            pred_c,
            sampling_rate=sampling_rate,
            channel_name=channel_name,
            baseline_preds=baseline_pred_c,
            baseline_name=baseline_name,
            model_name=model_name,
        )

    st.markdown("#### Residual Plot: Prediction Errors Over Time")
    render_residual_plot(
        true_c,
        pred_c,
        t_abs,
        channel_name,
        unit=unit,
        chunk_margin=chunk_margin,
        baseline_preds=baseline_pred_c,
        baseline_name=baseline_name,
        rescale=rescale,
    )

    render_statistics_table(
        true_c,
        pred_c,
        r_ch,
        channel_name,
        unit=unit,
        baseline_preds=baseline_pred_c,
        baseline_r=baseline_r,
        baseline_name=baseline_name,
        model_name=model_name,
        rescale=rescale,
    )

    st.markdown("---")
    with st.expander(diagnostics_label, expanded=False):
        render_residual_diagnostics(
            true_c,
            pred_c,
            channel_name,
            baseline_preds=baseline_pred_c,
            baseline_name=baseline_name,
            model_name=model_name,
            rescale=rescale,
        )
