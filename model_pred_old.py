import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, Dict, Any, List
import polars as pl
import pickle
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA as SklearnCCA
from umap import UMAP

from training.components.tester import Tester
from training.components.data import create_dataloaders
from dashboard.backbone import update_fig_title, PALETTE
from utils.config import get_config
from utils.polars import get_scalar_value, convert_series_to_list
from utils.stats import (
    compute_residual_statistics,
    qq_plot_data,
    normality_tests,
    probability_plot_data,
    compute_power_spectrum,
    find_dominant_frequencies,
    spectral_correlation,
    whiteness_test,
)
from dashboard.backbone import PALETTE

# TODO: maybe make this configurable?
resampled_freq = 60


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
    return sorted(list(ts))


def config_for_variant(project_root: Path, variant_name: str) -> Optional[Path]:
    cfg = project_root / "training" / "setups" / f"{variant_name}.yaml"
    return cfg if cfg.exists() else None


def check_precomputed_results(variant_dir: Path, run_ts: str) -> Dict[str, bool]:
    available = {}
    run_dir = variant_dir / run_ts

    for split in ["train", "val", "test"]:
        pickle_path = run_dir / f"{split}_results.pkl"
        parquet_path = variant_dir / f"{split}_results_{run_ts}"

        available[split] = pickle_path.exists() or parquet_path.exists()

    return available


def load_precomputed_results(
    variant_dir: Path, run_ts: str, split: str
) -> Optional[Dict[str, Any]]:

    run_dir = variant_dir / run_ts
    pickle_path = run_dir / f"{split}_results.pkl"

    if pickle_path.exists():
        try:
            with open(pickle_path, "rb") as f:
                results = pickle.load(f)
            print(f"Loaded cached {split} results from {pickle_path}")
            return results
        except Exception as e:
            print(f"Warning: Could not load pickle cache: {e}")

    results_path = variant_dir / f"{split}_results_{run_ts}"
    if not results_path.exists():
        return None

    try:
        df = pl.read_parquet(results_path)
        n_trials = len(df)
        cols = df.columns

        pearson_overall = get_scalar_value(df, "pearson_overall_mean")
        if pearson_overall is None:
            pearson_overall = get_scalar_value(df, "metric_pearson_r_mean")
        if pearson_overall is None:
            pearson_overall = np.nan

        pearson_overall_z = get_scalar_value(df, "pearson_overall_mean_Z")
        if pearson_overall_z is None:
            pearson_overall_z = get_scalar_value(df, "metric_pearson_r_mean_Z")
        if pearson_overall_z is None:
            pearson_overall_z = np.nan

        results = {
            "Y": convert_series_to_list(df["Y"].to_list()),
            "Yp": convert_series_to_list(df["Yp"].to_list()),
            "Z": convert_series_to_list(df["Z"].to_list()) if "Z" in cols else None,
            "Zp": (
                convert_series_to_list(df["Zp"].to_list())
                if "Zp" in cols
                else [None] * n_trials
            ),
            "Xp": convert_series_to_list(df["Xp"].to_list()),
            "pearson_per_channel": (
                convert_series_to_list(df["pearson_per_channel"].to_list())
                if "pearson_per_channel" in cols
                else (
                    convert_series_to_list(df["pearsonr_per_channel"].to_list())
                    if "pearsonr_per_channel" in cols
                    else [[np.nan]] * n_trials
                )
            ),
            "pearson_mean": (
                df["pearson_mean"].to_list()
                if "pearson_mean" in cols
                else (
                    df["pearsonr_mean"].to_list()
                    if "pearsonr_mean" in cols
                    else [np.nan] * n_trials
                )
            ),
            "pearson_overall_mean": pearson_overall,
            "pearson_per_channel_Z": (
                convert_series_to_list(df["pearson_per_channel_Z"].to_list())
                if "pearson_per_channel_Z" in cols
                else (
                    convert_series_to_list(df["pearsonr_per_channel_Z"].to_list())
                    if "pearsonr_per_channel_Z" in cols
                    else None
                )
            ),
            "pearson_mean_Z": (
                df["pearson_mean_Z"].to_list()
                if "pearson_mean_Z" in cols
                else (
                    df["pearsonr_mean_Z"].to_list()
                    if "pearsonr_mean_Z" in cols
                    else None
                )
            ),
            "pearson_overall_mean_Z": pearson_overall_z,
            "time": (
                convert_series_to_list(df["time"].to_list())
                if "time" in cols
                else [None] * n_trials
            ),
            "offset": (
                df["offset"].to_list() if "offset" in cols else [None] * n_trials
            ),
            "chunk_margin": (
                df["chunk_margin"].to_list()
                if "chunk_margin" in cols
                else [None] * n_trials
            ),
            "margined_duration": (
                df["margined_duration"].to_list()
                if "margined_duration" in cols
                else [None] * n_trials
            ),
            "stim": (df["stim"].to_list() if "stim" in cols else [None] * n_trials),
            "participant_id": df["participant_id"].to_list(),
            "session": df["session"].to_list(),
            "block": df["block"].to_list(),
            "trial": df["trial"].to_list(),
        }

        forecast_cols = [
            "Y_future_true",
            "Y_future_pred",
            "Y_concat_for_plot",
            "Z_future_true",
            "Z_future_pred",
            "Z_concat_for_plot",
            "X_future_pred",
        ]
        for fc in forecast_cols:
            if fc in cols:
                results[fc] = convert_series_to_list(df[fc].to_list())

        if "input_channels" in cols:
            ic_list = df["input_channels"].to_list()
            input_channels = ic_list[0] if len(ic_list) > 0 else []
            if isinstance(input_channels, pl.Series):
                input_channels = input_channels.to_list()
        else:
            input_channels = []
            for col in cols:
                if (
                    col.startswith(("ECOG_", "LFP_"))
                    and "_epochs" not in col
                    and "_psd" not in col
                ):
                    input_channels.append(col)
            input_channels = sorted(list(set(input_channels))) if input_channels else []
        results["input_channels"] = input_channels

        output_channels = []
        for col in cols:
            if col in [
                "tracing_velocity_magnitude",
                "tracing_velocity_x",
                "tracing_velocity_y",
                "x",
                "y",
            ]:
                output_channels.append(col)
        if not output_channels and "output_channels" in cols:
            oc_list = df["output_channels"].to_list()
            output_channels = oc_list[0] if len(oc_list) > 0 else []
            if isinstance(output_channels, pl.Series):
                output_channels = output_channels.to_list()
        results["output_channels"] = output_channels if output_channels else []

        return results
    except Exception as e:
        st.warning(f"Failed to load pre-computed results for {split}: {e}")
        return None


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


def render_y_prediction_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_abs: np.ndarray,
    channel_idx: int,
    channel_name: str,
    r_ch: float,
):
    n_chan = y_true.shape[1] if y_true.ndim == 2 else 1
    y_true = transpose_if_needed(y_true, len(t_abs))
    if y_pred is not None:
        y_pred = transpose_if_needed(y_pred, len(t_abs))
    y_true_c = y_true.squeeze() if n_chan == 1 else y_true[:, channel_idx]
    y_pred_c = (
        None
        if y_pred is None
        else (y_pred.squeeze() if n_chan == 1 else y_pred[:, channel_idx])
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_abs,
            y=y_true_c,
            name="Y_true (µV)",
            mode="lines",
        )
    )
    if y_pred_c is not None:
        fig.add_trace(
            go.Scatter(
                x=t_abs,
                y=y_pred_c,
                name="Y_pred (µV)",
                mode="lines",
            )
        )

    fig.update_layout(
        title=f"Y and Y_p — {channel_name} (r={r_ch:.3f})",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_y_scatter_plot(
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    channel_name: str,
    r_ch: float,
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_true_c,
            y=y_pred_c,
            mode="markers",
            name="Predictions",
            marker=dict(
                size=4,
                color="rgba(31, 119, 180, 0.6)",
                line=dict(width=0),
            ),
        )
    )

    min_val = min(np.min(y_true_c), np.min(y_pred_c))
    max_val = max(np.max(y_true_c), np.max(y_pred_c))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Identity (y=x)",
            line=dict(color="red", dash="dash", width=2),
        )
    )

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        y_true_c, y_pred_c
    )
    r_squared = r_value**2

    x_line = np.array([min_val, max_val])
    y_line = slope * x_line + intercept

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="OLS Fit",
            line=dict(color="blue", width=2),
        )
    )

    annotation_text = (
        f"<b>OLS Regression:</b><br>"
        f"y = {slope:.4f}x + {intercept:.4f}<br>"
        f"R² = {r_squared:.4f}<br>"
        f"p-value = {p_value:.2e}"
    )

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10),
        align="left",
        xanchor="left",
        yanchor="top",
    )

    fig.update_layout(
        title=f"True vs Predicted — {channel_name} (Pearson r={r_ch:.3f})",
        xaxis_title="Y_true (µV)",
        yaxis_title="Y_pred (µV)",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### OLS Regression Coefficients")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Slope", f"{slope:.4f}")
    with col2:
        st.metric("Intercept (µV)", f"{intercept:.4f}")
    with col3:
        st.metric("R²", f"{r_squared:.4f}")
    with col4:
        st.metric("p-value", f"{p_value:.2e}")


def render_y_residual_plot(
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    t_abs: np.ndarray,
    channel_name: str,
):
    residuals = y_pred_c - y_true_c

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t_abs,
            y=residuals,
            mode="lines",
            name="Residuals",
            line=dict(color=PALETTE.strawberry_red),
        )
    )
    rmse = np.sqrt(np.mean(residuals**2))

    fig.update_layout(
        title=f"Residuals (Prediction Errors) — {channel_name} (RMSE={rmse:.3f} µV)",
        xaxis_title="Time (s)",
        yaxis_title="Error (µV)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_statistics_table(
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    r_ch: float,
    channel_name: str,
):
    residuals = y_pred_c - y_true_c
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mse = np.mean(residuals**2)

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true_c - np.mean(y_true_c)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    st.markdown(f"### Statistics — {channel_name}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Pearson r", f"{r_ch:.4f}")

    with col2:
        st.metric("R²", f"{r_squared:.4f}")

    with col3:
        st.metric("RMSE (µV)", f"{rmse:.3f}")

    with col4:
        st.metric("MAE (µV)", f"{mae:.3f}")


def render_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    channel_name: str,
):

    res_stats = compute_residual_statistics(y_true, y_pred)
    residuals = res_stats["residuals"]

    st.markdown(f"### Residual Diagnostics — {channel_name}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{res_stats['mean']:.4f}")
    with col2:
        st.metric("Std Dev", f"{res_stats['std']:.4f}")
    with col3:
        st.metric("Min", f"{res_stats['min']:.4f}")
    with col4:
        st.metric("Max", f"{res_stats['max']:.4f}")

    norm_tests = normality_tests(residuals)

    st.markdown("#### Normality Tests")
    st.markdown(
        "**Null Hypothesis:** Residuals are normally distributed. "
        "Low p-values (< 0.05) suggest non-normality."
    )

    col1, col2 = st.columns(2)
    with col1:
        shapiro_stat, shapiro_p = norm_tests["shapiro"]
        st.metric(
            "Shapiro-Wilk Test",
            f"p = {shapiro_p:.4f}",
            delta=f"stat = {shapiro_stat:.4f}",
        )

    with col2:
        ks_stat, ks_p = norm_tests["ks"]
        st.metric(
            "Kolmogorov-Smirnov Test",
            f"p = {ks_p:.4f}",
            delta=f"stat = {ks_stat:.4f}",
        )

    st.markdown("#### Q-Q Plot (Quantile-Quantile)")
    st.markdown(
        "Points should lie on the diagonal line if residuals are normally distributed."
    )

    theoretical_q, sample_q = qq_plot_data(residuals)

    if len(theoretical_q) > 0:
        fig_qq = go.Figure()

        fig_qq.add_trace(
            go.Scatter(
                x=theoretical_q,
                y=sample_q,
                mode="markers",
                name="Residuals",
                marker=dict(size=4, color="rgba(31, 119, 180, 0.6)"),
            )
        )

        min_val = min(theoretical_q.min(), sample_q.min())
        max_val = max(theoretical_q.max(), sample_q.max())
        fig_qq.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Normal Distribution",
                line=dict(color="red", dash="dash", width=2),
            )
        )

        fig_qq.update_layout(
            title=f"Q-Q Plot — {channel_name}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            showlegend=True,
        )
        st.plotly_chart(fig_qq, use_container_width=True)
    else:
        st.warning("Not enough data for Q-Q plot")

    st.markdown("#### Probability Plot (CDF Comparison)")
    st.markdown(
        "Empirical CDF should closely match theoretical normal CDF if residuals are Gaussian."
    )

    theoretical_cdf, empirical_cdf = probability_plot_data(residuals)

    if len(theoretical_cdf) > 0:
        fig_prob = go.Figure()

        fig_prob.add_trace(
            go.Scatter(
                x=theoretical_cdf,
                y=empirical_cdf,
                mode="markers",
                name="Empirical CDF",
                marker=dict(size=3, color="rgba(31, 119, 180, 0.6)"),
            )
        )

        fig_prob.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect Match",
                line=dict(color="red", dash="dash", width=2),
            )
        )

        fig_prob.update_layout(
            title=f"Probability Plot — {channel_name}",
            xaxis_title="Theoretical CDF (Normal)",
            yaxis_title="Empirical CDF",
            showlegend=True,
        )
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.warning("Not enough data for probability plot")

    st.markdown("#### Whiteness Test (Ljung-Box)")
    whiteness_results = whiteness_test(residuals)
    lb_stat = whiteness_results["ljung_box_stat"]
    lb_p = whiteness_results["ljung_box_p"]
    lags = whiteness_results["lags"]
    acf = whiteness_results["acf"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Ljung-Box Statistic",
            f"{lb_stat:.4f}" if not np.isnan(lb_stat) else "N/A",
        )
    with col2:
        st.metric(
            "Ljung-Box p-value",
            f"{lb_p:.4f}" if not np.isnan(lb_p) else "N/A",
        )

    if len(lags) > 0 and len(acf) > 0:
        fig_acf = go.Figure()

        fig_acf.add_trace(
            go.Bar(
                x=lags,
                y=acf,
                name="ACF",
                marker=dict(color="rgba(31, 119, 180, 0.6)"),
            )
        )

        confidence_interval = 1.96 / np.sqrt(len(residuals.flatten()))
        fig_acf.add_hline(
            y=confidence_interval,
            line_dash="dash",
            line_color="red",
            annotation_text="95% CI",
        )
        fig_acf.add_hline(
            y=-confidence_interval,
            line_dash="dash",
            line_color="red",
        )

        fig_acf.update_layout(
            title=f"Autocorrelation Function — {channel_name}",
            xaxis_title="Lag",
            yaxis_title="ACF",
            showlegend=False,
        )
        st.plotly_chart(fig_acf, use_container_width=True)
    else:
        st.warning("Not enough data for ACF plot")


def render_learned_noise_diagnostics(
    config_path: str,
    run_ts: str,
):
    st.markdown("### Learned Noise Covariance Matrices")
    st.markdown(
        "These matrices characterize the stochastic dynamics learned by the model:\n"
        "- **Q**: Process noise covariance (state dynamics)\n"
        "- **R**: Observation noise covariance (measurement noise)\n"
        "- **S**: Cross-covariance between process and observation noise"
    )

    with st.spinner("Loading model..."):
        tester = Tester.from_config_file(config_path, run_timestamp=run_ts)
        tester._load_model_for_run()

    model = tester.framework.model

    if hasattr(model, "idSys"):
        idSys = model.idSys

        Q = getattr(idSys, "Q", None)
        R = getattr(idSys, "R", None)
        S = getattr(idSys, "S", None)

        if Q is not None or R is not None or S is not None:
            cols = st.columns(3)

            if Q is not None:
                with cols[0]:
                    st.markdown("#### Process Noise (Q)")
                    Q_np = np.array(Q) if not isinstance(Q, np.ndarray) else Q
                    fig_q = go.Figure(
                        data=go.Heatmap(
                            z=Q_np,
                            colorscale="RdBu_r",
                            zmid=0,
                            colorbar=dict(title="Value"),
                        )
                    )
                    fig_q.update_layout(
                        title="Q Matrix",
                        xaxis_title="State Dimension",
                        yaxis_title="State Dimension",
                        height=400,
                    )
                    st.plotly_chart(fig_q, use_container_width=True)
                    st.caption(f"Shape: {Q_np.shape}")

            if R is not None:
                with cols[1]:
                    st.markdown("#### Observation Noise (R)")
                    R_np = np.array(R) if not isinstance(R, np.ndarray) else R
                    fig_r = go.Figure(
                        data=go.Heatmap(
                            z=R_np,
                            colorscale="RdBu_r",
                            zmid=0,
                            colorbar=dict(title="Value"),
                        )
                    )
                    fig_r.update_layout(
                        title="R Matrix",
                        xaxis_title="Output Dimension",
                        yaxis_title="Output Dimension",
                        height=400,
                    )
                    st.plotly_chart(fig_r, use_container_width=True)
                    st.caption(f"Shape: {R_np.shape}")

            if S is not None:
                with cols[2]:
                    st.markdown("#### Cross-Covariance (S)")
                    S_np = np.array(S) if not isinstance(S, np.ndarray) else S
                    fig_s = go.Figure(
                        data=go.Heatmap(
                            z=S_np,
                            colorscale="RdBu_r",
                            zmid=0,
                            colorbar=dict(title="Value"),
                        )
                    )
                    fig_s.update_layout(
                        title="S Matrix",
                        xaxis_title="Output Dimension",
                        yaxis_title="State Dimension",
                        height=400,
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
                    st.caption(f"Shape: {S_np.shape}")
        else:
            st.info("No noise covariance matrices found in model")
    else:
        st.info("This model type does not expose noise covariance matrices")


def render_z_prediction_plot(
    z_true: np.ndarray,
    z_pred: np.ndarray,
    t_abs: np.ndarray,
    channel_idx: int,
    channel_name: str,
    r_ch: float,
):
    nz_chan = z_true.shape[1] if z_true.ndim == 2 else 1
    z_true = transpose_if_needed(z_true, len(t_abs))
    z_pred = transpose_if_needed(z_pred, len(t_abs))

    z_true_c = z_true.squeeze() if nz_chan == 1 else z_true[:, channel_idx]
    z_pred_c = z_pred.squeeze() if nz_chan == 1 else z_pred[:, channel_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_abs,
            y=z_true_c,
            name="Z_true",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_abs,
            y=z_pred_c,
            name="Z_pred",
            mode="lines",
        )
    )

    fig.update_layout(
        title=f"Z and Z_p — {channel_name} (r={r_ch:.3f})",
        xaxis_title="Time (s)",
        yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_predictions_tab(split_res: Dict[str, Any], trial_idx: int, cfg_path: Path):
    Y_true = split_res["Y"]
    Yp = split_res["Yp"]
    Zp = split_res["Zp"]
    pearson_tr = split_res["pearson_per_channel"]
    pearson_mean = split_res["pearson_mean"]

    y_t = np.array(Y_true[trial_idx])
    y_p = np.array(Yp[trial_idx])
    z_p = None if Zp[trial_idx] is None else np.array(Zp[trial_idx])

    r_list = pearson_tr[trial_idx] if pearson_tr else []

    if r_list:
        valid_r = [r for r in r_list if not (r is None or np.isnan(r))]
        r_mean = np.mean(valid_r) if len(valid_r) > 0 else np.nan
    else:
        r_mean = np.nan
    # TODO: fix it later
    # mean_str = f"{r_mean:.4f}" if not np.isnan(r_mean) else "nan"
    # st.markdown(f"**Pearson per channel:** {r_list} | **Mean:** {mean_str}")

    offsets = split_res.get("offset", [])
    t_offset = (
        float(offsets[trial_idx])
        if offsets and len(offsets) > trial_idx and offsets[trial_idx] is not None
        else 0.0
    )
    n_samples = y_t.shape[0]
    t_abs = get_trial_time_axis(split_res, trial_idx, n_samples, t_offset)

    chan_names = split_res.get("input_channels", [])
    n_chan = y_t.shape[1] if y_t.ndim == 2 else 1
    if chan_names and isinstance(chan_names, list) and len(chan_names) == n_chan:
        channel_options = chan_names
    else:
        channel_options = [f"ch{i}" for i in range(n_chan)]

    st.subheader("Neural Signal Predictions")
    selected_name = st.selectbox(
        "Channel for Y/Yp plot",
        options=channel_options,
        index=0,
        key="pred_chan",
    )
    c = channel_options.index(selected_name) if n_chan > 1 else 0
    r_ch = r_list[c] if r_list and c < len(r_list) else np.nan

    n_chan = y_t.shape[1] if y_t.ndim == 2 else 1
    y_t = transpose_if_needed(y_t, len(t_abs))
    y_p = transpose_if_needed(y_p, len(t_abs))

    y_true_c = y_t.squeeze() if n_chan == 1 else y_t[:, c]
    y_pred_c = y_p.squeeze() if n_chan == 1 else y_p[:, c]

    if y_true_c is not None and y_pred_c is not None and len(y_true_c) > 1:
        try:
            yt_flat = y_true_c.flatten()
            yp_flat = y_pred_c.flatten()
            if len(yt_flat) == len(yp_flat):
                r_calc = np.corrcoef(yt_flat, yp_flat)[0, 1]
                if not np.isnan(r_calc):
                    r_ch = r_calc
        except Exception:
            pass

    st.markdown("#### Time Series: Y_true vs Y_pred")
    fig_ts = go.Figure()
    fig_ts.add_trace(
        go.Scatter(
            x=t_abs,
            y=y_true_c,
            name="Y_true (µV)",
            mode="lines",
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=t_abs,
            y=y_pred_c,
            name="Y_pred (µV)",
            mode="lines",
        )
    )
    fig_ts.update_layout(
        title=f"Y and Y_p — {selected_name} (r={r_ch:.3f})",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("#### Scatter Plot: True vs Predicted")
    render_y_scatter_plot(y_true_c, y_pred_c, selected_name, r_ch)

    st.markdown("#### Residual Plot: Prediction Errors Over Time")
    render_y_residual_plot(y_true_c, y_pred_c, t_abs, selected_name)

    render_statistics_table(y_true_c, y_pred_c, r_ch, selected_name)

    st.markdown("---")
    with st.expander("Residual Diagnostics & Normality Tests", expanded=False):
        st.markdown(
            "Comprehensive diagnostics to verify that residuals follow a Gaussian distribution, "
            "which is a key assumption for many state-space models."
        )
        render_residual_diagnostics(y_true_c, y_pred_c, selected_name)

    st.markdown("---")
    with st.expander("Learned Noise Covariance Matrices", expanded=False):
        st.markdown(
            "Visualize the noise covariance matrices (Q, R, S) learned by the model during training. "
            "These characterize the stochastic dynamics of the system."
        )
        if "config_path" in st.session_state and "run_timestamp" in st.session_state:
            render_learned_noise_diagnostics(
                str(st.session_state["config_path"]), st.session_state["run_timestamp"]
            )
        else:
            st.info("Model configuration not available in session state")

    Z_true = split_res.get("Z")
    if Z_true is not None and z_p is not None:
        z_t = None if Z_true[trial_idx] is None else np.array(Z_true[trial_idx])

        if z_t is not None:
            st.subheader("Behavioral Variable Predictions")
            nz_chan = z_t.shape[1] if z_t.ndim == 2 else 1

            output_chan_names = split_res.get("output_channels", [])
            if (
                output_chan_names
                and isinstance(output_chan_names, list)
                and len(output_chan_names) == nz_chan
            ):
                z_channel_options = output_chan_names
            else:
                z_channel_options = [f"z_ch{i}" for i in range(nz_chan)]

            selected_z_name = st.selectbox(
                "Channel for Z/Zp plot",
                options=z_channel_options,
                index=0,
                key="pred_z_chan",
            )
            z_c = z_channel_options.index(selected_z_name) if nz_chan > 1 else 0

            pearson_z_tr = split_res.get("pearson_per_channel_Z", [])
            r_z_list = pearson_z_tr[trial_idx] if pearson_z_tr else []

            if r_z_list:
                valid_r_z = [r for r in r_z_list if not (r is None or np.isnan(r))]
                r_z_mean = np.mean(valid_r_z) if len(valid_r_z) > 0 else np.nan
            else:
                r_z_mean = np.nan

            r_z_ch = r_z_list[z_c] if r_z_list and z_c < len(r_z_list) else np.nan

            mean_z_str = f"{r_z_mean:.4f}" if not np.isnan(r_z_mean) else "nan"
            st.markdown(
                f"**Pearson per channel (Z):** {r_z_list} | **Mean:** {mean_z_str}"
            )

            render_z_prediction_plot(z_t, z_p, t_abs, z_c, selected_z_name, r_z_ch)


def render_y_forecast_plot(
    y_concat: np.ndarray,
    y_future_true: np.ndarray,
    y_future_pred: np.ndarray,
    t_abs_margined: np.ndarray,
    m_samples: int,
    channel_idx: int,
    channel_name: str,
    r_fore_ch: float,
):
    n_chan = y_concat.shape[1] if y_concat.ndim == 2 else 1
    y_concat_c = y_concat.squeeze() if n_chan == 1 else y_concat[:, channel_idx]
    y_ft_c = y_future_true.squeeze() if n_chan == 1 else y_future_true[:, channel_idx]
    y_fp_c = y_future_pred.squeeze() if n_chan == 1 else y_future_pred[:, channel_idx]

    T = len(y_concat_c)
    Tpast = max(0, T - m_samples)
    t_past = t_abs_margined[:Tpast]
    t_future = t_abs_margined[Tpast:T]

    t_present = (
        t_abs_margined[Tpast] if Tpast < len(t_abs_margined) else t_abs_margined[-1]
    )

    fig = go.Figure()

    if Tpast > 0:
        fig.add_trace(
            go.Scatter(
                x=t_past,
                y=y_concat_c[:Tpast],
                name="History",
                mode="lines",
                line=dict(color=PALETTE.twilight_indigo, width=2),
            )
        )

        t_future_plot = t_abs_margined[Tpast - 1 : T]
        last_hist_val = y_concat_c[Tpast - 1]
        y_ft_plot = np.concatenate(([last_hist_val], y_ft_c))
        y_fp_plot = np.concatenate(([last_hist_val], y_fp_c))
    else:
        t_future_plot = t_future
        y_ft_plot = y_ft_c
        y_fp_plot = y_fp_c

    fig.add_trace(
        go.Scatter(
            x=t_future_plot,
            y=y_ft_plot,
            name="True Future",
            mode="lines",
            line=dict(color="#2ca02c", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t_future_plot,
            y=y_fp_plot,
            name="Forecast",
            mode="lines",
            line=dict(color=PALETTE.strawberry_red, width=2),
        )
    )

    fig.add_vline(
        x=t_present,
        line_dash="dash",
        line_color="gray",
        line_width=2,
        annotation_text="Present",
        annotation_position="top",
    )

    m_seconds = m_samples / resampled_freq
    fig.update_layout(
        title=f"Y Forecast (horizon={m_seconds:.2f}s) — {channel_name} (r={r_fore_ch:.3f})",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_z_forecast_plot(
    z_concat: Optional[np.ndarray],
    z_future_true: np.ndarray,
    z_future_pred: np.ndarray,
    t_abs_margined: np.ndarray,
    m_samples: int,
    channel_idx: int,
    channel_name: str,
    r_fore_z_ch: float,
):

    nz_chan = z_future_true.shape[1] if z_future_true.ndim == 2 else 1

    z_ft_c = z_future_true.squeeze() if nz_chan == 1 else z_future_true[:, channel_idx]
    z_fp_c = z_future_pred.squeeze() if nz_chan == 1 else z_future_pred[:, channel_idx]

    T = (
        len(z_concat)
        if z_concat is not None
        else len(z_ft_c) + len(t_abs_margined) - m_samples
    )
    Tpast = max(0, len(t_abs_margined) - m_samples)
    t_past = t_abs_margined[:Tpast]
    t_future = t_abs_margined[Tpast : Tpast + len(z_ft_c)]

    t_present = (
        t_abs_margined[Tpast] if Tpast < len(t_abs_margined) else t_abs_margined[-1]
    )

    fig = go.Figure()

    if z_concat is not None:
        z_concat = np.array(z_concat)
        z_concat_c = z_concat.squeeze() if nz_chan == 1 else z_concat[:, channel_idx]
        if Tpast > 0 and len(z_concat_c) >= Tpast:
            fig.add_trace(
                go.Scatter(
                    x=t_past,
                    y=z_concat_c[:Tpast],
                    name="History",
                    mode="lines",
                    line=dict(color=PALETTE.twilight_indigo, width=2),
                )
            )

            t_future_plot = t_abs_margined[Tpast - 1 : Tpast + len(z_ft_c)]
            last_hist_val = z_concat_c[Tpast - 1]
            z_ft_plot = np.concatenate(([last_hist_val], z_ft_c))
            z_fp_plot = np.concatenate(([last_hist_val], z_fp_c))
        else:
            t_future_plot = t_future
            z_ft_plot = z_ft_c
            z_fp_plot = z_fp_c
    else:
        t_future_plot = t_future
        z_ft_plot = z_ft_c
        z_fp_plot = z_fp_c

    fig.add_trace(
        go.Scatter(
            x=t_future_plot,
            y=z_ft_plot,
            name="True Future",
            mode="lines",
            line=dict(color="#2ca02c", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_future_plot,
            y=z_fp_plot,
            name="Forecast",
            mode="lines",
            line=dict(color=PALETTE.strawberry_red, width=2),
        )
    )

    fig.add_vline(
        x=t_present,
        line_dash="dash",
        line_color="gray",
        line_width=2,
        annotation_text="Present",
        annotation_position="top",
    )

    m_seconds = m_samples / resampled_freq
    fig.update_layout(
        title=f"Z Forecast (horizon={m_seconds:.2f}s) — {channel_name} (r={r_fore_z_ch:.3f})",
        xaxis_title="Time (s)",
        yaxis_title="Value",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_forecasting_tab(
    split_res: Dict[str, Any],
    trial_idx: int,
    cfg_path: Path,
    run_ts: str,
    Y_true: List,
    Yp: List,
):

    f_res = None

    if "Y_future_pred" in split_res and split_res["Y_future_pred"] is not None:
        if len(split_res["Y_future_pred"]) > trial_idx:
            f_res = {}
            keys_to_copy = [
                "Y_future_true",
                "Y_future_pred",
                "Y_concat_for_plot",
                "Z_future_true",
                "Z_future_pred",
                "Z_concat_for_plot",
                "X_future_pred",
                "pearson_per_channel",
                "pearson_per_channel_Z",
            ]
            for k in keys_to_copy:
                if k in split_res:
                    val = split_res[k][trial_idx]
                    if val is not None:
                        f_res[k] = val

            if "metric_m" in split_res:
                m_val = split_res["metric_m"]
                if isinstance(m_val, list) and len(m_val) > 0:
                    f_res["m"] = m_val[0]
                else:
                    f_res["m"] = m_val
            else:
                try:
                    from utils.config import get_config

                    cfg = get_config(str(cfg_path))
                    m_seconds = cfg.model.forecast.m
                    sampling_freq = cfg.data.sampling_frequency
                    f_res["m"] = int(m_seconds * sampling_freq)
                except Exception:
                    f_res["m"] = 0

    if f_res is None and "trial_forecasts" in split_res:
        f_res = split_res["trial_forecasts"].get(trial_idx)

    if f_res is None:
        with st.spinner(f"Computing forecast for trial {trial_idx}..."):
            y_trial = np.array(Y_true[trial_idx])
            z_trial = (
                np.array(split_res.get("Z", [None])[trial_idx])
                if split_res.get("Z") and split_res["Z"][trial_idx] is not None
                else None
            )
            chunk_margin = (
                split_res["chunk_margin"][trial_idx]
                if split_res.get("chunk_margin")
                else None
            )

            trial_forecast = compute_forecast_for_trial(
                str(cfg_path),
                run_ts,
                y_trial,
                z_trial,
                chunk_margin,
            )

            if "trial_forecasts" not in split_res:
                split_res["trial_forecasts"] = {}
            split_res["trial_forecasts"][trial_idx] = trial_forecast
            f_res = trial_forecast

    if not f_res:
        st.warning("Failed to compute forecast for this trial.")
        return

    try:
        m = int(f_res.get("m", 0))
        margin_samples = int(f_res.get("margin_samples", 0))

        y_concat = f_res.get("Y_concat_for_plot")
        y_future_true = f_res.get("Y_future_true")
        y_future_pred = f_res.get("Y_future_pred")
        r_fore_list = f_res.get("pearson_per_channel")

        if (
            y_concat is not None
            and y_future_true is not None
            and y_future_pred is not None
            and m > 0
        ):
            y_concat = np.array(y_concat)
            y_future_true = np.array(y_future_true)
            y_future_pred = np.array(y_future_pred)

            meta_time_margined = split_res.get("time_margined", [])
            md_list = split_res.get("margined_duration", [])
            offsets = split_res.get("offset", [])

            if meta_time_margined and len(meta_time_margined) > trial_idx:
                t_full = np.array(meta_time_margined[trial_idx])
                if len(t_full) >= len(y_concat):
                    t_abs_margined = t_full[: len(y_concat)]
                else:
                    t_abs_margined = np.linspace(
                        0.0, len(y_concat) / resampled_freq, len(y_concat)
                    )
            else:
                t_abs_margined = np.linspace(
                    0.0, len(y_concat) / resampled_freq, len(y_concat)
                )

            t_offset = (
                float(offsets[trial_idx])
                if offsets
                and len(offsets) > trial_idx
                and offsets[trial_idx] is not None
                else 0.0
            )
            t_abs_margined = t_abs_margined + t_offset

            chan_names = split_res.get("input_channels", [])
            n_chan = y_concat.shape[1] if y_concat.ndim == 2 else 1
            if (
                chan_names
                and isinstance(chan_names, list)
                and len(chan_names) == n_chan
            ):
                channel_options = chan_names
            else:
                channel_options = [f"ch{i}" for i in range(n_chan)]

            st.subheader("Neural Signal Forecast")
            selected_name = st.selectbox(
                "Channel for Y forecast plot",
                options=channel_options,
                index=0,
                key="forecast_chan",
            )
            c = channel_options.index(selected_name) if n_chan > 1 else 0

            r_fore_ch = np.nan
            if (
                r_fore_list is not None
                and isinstance(r_fore_list, (list, tuple))
                and len(r_fore_list) > c
            ):
                r_fore_ch = r_fore_list[c]

            render_y_forecast_plot(
                y_concat,
                y_future_true,
                y_future_pred,
                t_abs_margined,
                m,
                c,
                selected_name,
                r_fore_ch,
            )

            st.markdown("---")
            try:
                cfg = get_config(str(cfg_path))
                sampling_freq = cfg.data.sampling_frequency

                y_true_ch = (
                    y_future_true[:, c] if y_future_true.ndim == 2 else y_future_true
                )
                y_pred_ch = (
                    y_future_pred[:, c] if y_future_pred.ndim == 2 else y_future_pred
                )

                render_frequency_analysis(
                    y_true_ch,
                    y_pred_ch,
                    selected_name,
                    sampling_freq,
                )
            except Exception as e:
                st.warning(f"Could not render frequency analysis: {e}")

            z_concat = f_res.get("Z_concat_for_plot")
            z_future_true = f_res.get("Z_future_true")
            z_future_pred = f_res.get("Z_future_pred")
            r_fore_list_z = f_res.get("pearson_per_channel_Z")

            if z_future_true is not None and z_future_pred is not None and m > 0:
                try:
                    z_future_true = np.array(z_future_true)
                    z_future_pred = np.array(z_future_pred)

                    st.subheader("Behavioral Variable Forecast")
                    nz_chan = z_future_true.shape[1] if z_future_true.ndim == 2 else 1

                    output_chan_names = split_res.get("output_channels", [])
                    if (
                        output_chan_names
                        and isinstance(output_chan_names, list)
                        and len(output_chan_names) == nz_chan
                    ):
                        z_channel_options = output_chan_names
                    else:
                        z_channel_options = [f"z_ch{i}" for i in range(nz_chan)]

                    selected_z_name = st.selectbox(
                        "Channel for Z forecast plot",
                        options=z_channel_options,
                        index=0,
                        key="forecast_z_chan",
                    )
                    z_c = z_channel_options.index(selected_z_name) if nz_chan > 1 else 0

                    r_fore_z_ch = np.nan
                    if r_fore_list_z is not None:
                        if (
                            isinstance(r_fore_list_z, (list, tuple, np.ndarray))
                            and len(r_fore_list_z) > z_c
                        ):
                            r_fore_z_ch = r_fore_list_z[z_c]

                    render_z_forecast_plot(
                        z_concat,
                        z_future_true,
                        z_future_pred,
                        t_abs_margined,
                        m,
                        z_c,
                        selected_z_name,
                        r_fore_z_ch,
                    )
                except Exception as e:
                    st.warning(f"Could not render Z forecast: {e}")

            x_future_pred = f_res.get("X_future_pred")

            if x_future_pred is not None:
                try:
                    x_future_pred = np.array(x_future_pred)

                    Xp = split_res.get("Xp", [])
                    if Xp and len(Xp) > trial_idx and Xp[trial_idx] is not None:
                        x_p_trial = np.array(Xp[trial_idx])

                        if len(x_p_trial) >= m:
                            x_future_true = x_p_trial[-m:]

                            st.markdown("---")
                            st.subheader("Latent States Forecast Frequency Analysis")
                            st.markdown(
                                "Verify that forecasted latent states preserve frequency dynamics"
                            )

                            if x_future_true.ndim == 1:
                                x_future_true = x_future_true.reshape(-1, 1)
                            if x_future_pred.ndim == 1:
                                x_future_pred = x_future_pred.reshape(-1, 1)

                            if x_future_true.shape[0] < x_future_true.shape[1]:
                                x_future_true = x_future_true.T
                            if x_future_pred.shape[0] < x_future_pred.shape[1]:
                                x_future_pred = x_future_pred.T

                            n_latent_dims = min(
                                x_future_true.shape[1], x_future_pred.shape[1]
                            )

                            latent_dim = st.selectbox(
                                "Latent dimension for frequency analysis",
                                options=list(range(n_latent_dims)),
                                format_func=lambda x: f"Dimension {x+1}",
                                key="forecast_latent_dim",
                            )

                            x_true_dim = x_future_true[:, latent_dim]
                            x_pred_dim = x_future_pred[:, latent_dim]

                            min_len = min(len(x_true_dim), len(x_pred_dim))
                            x_true_dim = x_true_dim[:min_len]
                            x_pred_dim = x_pred_dim[:min_len]

                            cfg = get_config(str(cfg_path))
                            sampling_freq = cfg.data.sampling_frequency

                            render_frequency_analysis(
                                x_true_dim,
                                x_pred_dim,
                                f"Latent Dimension {latent_dim+1}",
                                sampling_freq,
                            )

                except Exception as e:
                    st.warning(
                        f"Could not render latent forecast frequency analysis: {e}"
                    )

    except Exception as e:
        st.warning(f"Could not render forecast plot: {e}")


def render_frequency_analysis(
    y_future_true: np.ndarray,
    y_future_pred: np.ndarray,
    channel_name: str,
    sampling_freq: float = 1000.0,
):

    st.markdown(f"### Frequency Analysis — {channel_name}")
    st.markdown("Compare frequency content of true vs predicted signals")

    freqs_true, psd_true = compute_power_spectrum(y_future_true, sampling_freq)
    freqs_pred, psd_pred = compute_power_spectrum(y_future_pred, sampling_freq)

    dom_freqs_true, dom_powers_true = find_dominant_frequencies(
        freqs_true, psd_true, n_peaks=5
    )
    dom_freqs_pred, dom_powers_pred = find_dominant_frequencies(
        freqs_pred, psd_pred, n_peaks=5
    )

    spec_corr = spectral_correlation(freqs_true, psd_true, psd_pred)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=freqs_true,
            y=psd_true,
            mode="lines",
            name="True Signal",
            line=dict(color="blue", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=freqs_pred,
            y=psd_pred,
            mode="lines",
            name="Predicted Signal",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    if len(dom_freqs_true) > 0:
        fig.add_trace(
            go.Scatter(
                x=dom_freqs_true,
                y=dom_powers_true,
                mode="markers",
                name="True Peaks",
                marker=dict(size=10, color="blue", symbol="x"),
            )
        )

    if len(dom_freqs_pred) > 0:
        fig.add_trace(
            go.Scatter(
                x=dom_freqs_pred,
                y=dom_powers_pred,
                mode="markers",
                name="Predicted Peaks",
                marker=dict(size=10, color="red", symbol="circle"),
            )
        )

    fig.update_layout(
        title=f"Power Spectrum Comparison — {channel_name}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density",
        yaxis_type="log",
        showlegend=True,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Dominant Frequencies (True)")
        if len(dom_freqs_true) > 0:
            for i, (freq, power) in enumerate(zip(dom_freqs_true, dom_powers_true)):
                st.metric(f"Peak {i+1}", f"{freq:.2f} Hz", f"Power: {power:.2e}")
        else:
            st.info("No dominant peaks detected")

    with col2:
        st.markdown("#### Dominant Frequencies (Predicted)")
        if len(dom_freqs_pred) > 0:
            for i, (freq, power) in enumerate(zip(dom_freqs_pred, dom_powers_pred)):
                st.metric(f"Peak {i+1}", f"{freq:.2f} Hz", f"Power: {power:.2e}")
        else:
            st.info("No dominant peaks detected")

    st.markdown("#### Spectral Correlation")
    st.metric(
        "Correlation between spectra",
        f"{spec_corr:.4f}",
    )


def render_latent_states_plot(
    x_p: np.ndarray,
    t_abs: np.ndarray,
    t_offset: float,
    chunk_margin: Optional[float],
    duration: Optional[float],
    trial_idx: int,
):
    fig = go.Figure()
    t_x = (
        np.linspace(t_abs[0], t_abs[-1], x_p.shape[0])
        if len(t_abs) != x_p.shape[0]
        else t_abs
    )
    nx = x_p.shape[1] if x_p.ndim == 2 else 1
    x_min = float(np.nanmin(x_p))
    x_max = float(np.nanmax(x_p))

    for d in range(nx):
        series = x_p[:, d] if nx > 1 else x_p.squeeze()
        fig.add_trace(
            go.Scatter(
                x=t_x,
                y=series,
                name=f"X_p[{d}]",
                mode="lines",
            )
        )

    if duration is not None:
        event_start = (
            t_offset + float(chunk_margin) if chunk_margin is not None else t_x[0]
        )
        event_end = (
            t_offset
            + float(duration)
            - (float(chunk_margin) if chunk_margin is not None else 0.0)
        )
        fig.add_vrect(
            x0=event_start,
            x1=event_end,
            fillcolor="rgba(0, 100, 0, 0.1)",
            layer="below",
            line_width=0,
        )
        fig.add_vline(x=event_start, line_dash="dash", line_color="green")
        fig.add_vline(x=event_end, line_dash="dash", line_color="red")

    fig.update_layout(
        title=f"Latent states X_p — Trial {trial_idx}",
        xaxis_title="Time (s)",
        yaxis_title="Raw value",
        xaxis_range=[t_x[0], t_x[-1]],
        yaxis_range=[x_min, x_max],
    )
    st.plotly_chart(fig, use_container_width=True)


def render_auxiliary_predictions_plot(
    z_p: np.ndarray,
    t_abs: np.ndarray,
    t_offset: float,
    chunk_margin: Optional[float],
    duration: Optional[float],
    trial_idx: int,
):
    fig = go.Figure()
    t_z = (
        np.linspace(t_abs[0], t_abs[-1], z_p.shape[0])
        if len(t_abs) != z_p.shape[0]
        else t_abs
    )
    nz = z_p.shape[1] if z_p.ndim == 2 else 1
    z_min = float(np.nanmin(z_p))
    z_max = float(np.nanmax(z_p))

    for d in range(nz):
        series = z_p[:, d] if nz > 1 else z_p.squeeze()
        fig.add_trace(
            go.Scatter(
                x=t_z,
                y=series,
                name=f"Z_p[{d}]",
                mode="lines",
            )
        )

    if duration is not None:
        event_start = (
            t_offset + float(chunk_margin) if chunk_margin is not None else t_z[0]
        )
        event_end = (
            t_offset
            + float(duration)
            - (float(chunk_margin) if chunk_margin is not None else 0.0)
        )
        fig.add_vrect(
            x0=event_start,
            x1=event_end,
            fillcolor="rgba(0, 100, 0, 0.1)",
            layer="below",
            line_width=0,
        )
        fig.add_vline(x=event_start, line_dash="dash", line_color="green")
        fig.add_vline(x=event_end, line_dash="dash", line_color="red")

    fig.update_layout(
        title=f"Other predictions Z_p — Trial {trial_idx}",
        xaxis_title="Time (s)",
        yaxis_title="Value",
        xaxis_range=[t_z[0], t_z[-1]],
        yaxis_range=[z_min, z_max],
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tsne_umap_plot(
    x_p: np.ndarray, trial_idx: int, z_p: Optional[np.ndarray] = None
):
    from sklearn.manifold import TSNE
    from umap import UMAP

    if x_p.ndim != 2:
        st.warning("Latent states must be 2D for dimensionality reduction.")
        return

    n_samples, n_dims = x_p.shape

    if n_dims < 3:
        st.info(
            f"Latent space has only {n_dims} dimensions. Dimensionality reduction is most useful for higher-dimensional spaces."
        )
        return

    if z_p is not None and z_p.size > 0:
        if z_p.ndim == 2:
            color_values = z_p[:, 0]
            color_label = "Tracing Speed"
        else:
            color_values = z_p.squeeze()
            color_label = "Tracing Speed"

        if len(color_values) != n_samples:
            if len(color_values) > n_samples:
                color_values = color_values[:n_samples]
            else:
                color_values = np.arange(n_samples)
                color_label = "Time"
    else:
        color_values = np.arange(n_samples)
        color_label = "Time"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### t-SNE Projection")
        with st.spinner("Computing t-SNE..."):
            perplexity = min(30, n_samples - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            x_tsne = tsne.fit_transform(x_p)

        fig_tsne = go.Figure()
        fig_tsne.add_trace(
            go.Scatter(
                x=x_tsne[:, 0],
                y=x_tsne[:, 1],
                mode="markers",
                marker=dict(
                    size=4,
                    color=color_values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=color_label),
                ),
                text=[
                    f"t={i}, {color_label.lower()}={color_values[i]:.3f}"
                    for i in range(n_samples)
                ],
                hovertemplate="<b>Time step:</b> %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
            )
        )
        fig_tsne.update_layout(
            title=f"t-SNE of Latent States — Trial {trial_idx}",
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            height=400,
        )
        st.plotly_chart(fig_tsne, use_container_width=True)

    with col2:
        st.markdown("##### UMAP Projection")
        with st.spinner("Computing UMAP..."):
            n_neighbors = min(15, n_samples - 1)
            umap = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
            x_umap = umap.fit_transform(x_p)

        fig_umap = go.Figure()
        fig_umap.add_trace(
            go.Scatter(
                x=x_umap[:, 0],
                y=x_umap[:, 1],
                mode="markers",
                marker=dict(
                    size=4,
                    color=color_values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=color_label),
                ),
                text=[
                    f"t={i}, {color_label.lower()}={color_values[i]:.3f}"
                    for i in range(n_samples)
                ],
                hovertemplate="<b>Time step:</b> %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
            )
        )
        fig_umap.update_layout(
            title=f"UMAP of Latent States — Trial {trial_idx}",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            height=400,
        )
        st.plotly_chart(fig_umap, use_container_width=True)


def render_cca_analysis(
    x_p: np.ndarray,
    z_data: np.ndarray,
    trial_idx: int,
    x_label: str = "Latent States (X)",
    z_label: str = "Behavioral Variable (Z)",
):

    if x_p.ndim != 2 or z_data.ndim != 2:
        st.warning("Both datasets must be 2D for CCA analysis.")
        return

    n_samples_x, n_features_x = x_p.shape
    n_samples_z, n_features_z = z_data.shape

    if n_samples_x != n_samples_z:
        st.warning(
            f"Sample size mismatch: {x_label} has {n_samples_x} samples, {z_label} has {n_samples_z} samples."
        )
        return

    n_samples = n_samples_x

    n_components = min(n_features_x, n_features_z, n_samples - 1)

    if n_components < 1:
        st.warning("Not enough components for CCA analysis.")
        return

    st.markdown(f"##### Canonical Correlation Analysis")
    st.markdown(
        f"Analyzing relationship between **{x_label}** ({n_features_x} dims) and **{z_label}** ({n_features_z} dims)"
    )

    with st.spinner("Computing CCA..."):
        cca = SklearnCCA(n_components=n_components)

        x_c, z_c = cca.fit_transform(x_p, z_data)

        canonical_corrs = np.array(
            [np.corrcoef(x_c[:, i], z_c[:, i])[0, 1] for i in range(n_components)]
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Canonical Correlations**")
        fig_corr = go.Figure()
        fig_corr.add_trace(
            go.Bar(
                x=list(range(1, n_components + 1)),
                y=canonical_corrs,
                marker_color="steelblue",
                text=[f"{corr:.3f}" for corr in canonical_corrs],
                textposition="outside",
            )
        )
        fig_corr.update_layout(
            title=f"Canonical Correlations — Trial {trial_idx}",
            xaxis_title="Canonical Component",
            yaxis_title="Correlation",
            yaxis_range=[0, 1.1],
            height=400,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.markdown("**Cumulative Variance Explained**")
        var_x = np.var(x_c, axis=0)
        var_z = np.var(z_c, axis=0)

        var_x_prop = var_x / np.sum(var_x) * 100
        var_z_prop = var_z / np.sum(var_z) * 100

        cum_var_x = np.cumsum(var_x_prop)
        cum_var_z = np.cumsum(var_z_prop)

        fig_var = go.Figure()
        fig_var.add_trace(
            go.Scatter(
                x=list(range(1, n_components + 1)),
                y=cum_var_x,
                mode="lines+markers",
                name=x_label,
                line=dict(color="blue", width=2),
                marker=dict(size=8),
            )
        )
        fig_var.add_trace(
            go.Scatter(
                x=list(range(1, n_components + 1)),
                y=cum_var_z,
                mode="lines+markers",
                name=z_label,
                line=dict(color="red", width=2),
                marker=dict(size=8),
            )
        )
        fig_var.update_layout(
            title=f"Cumulative Variance — Trial {trial_idx}",
            xaxis_title="Canonical Component",
            yaxis_title="Cumulative Variance (%)",
            yaxis_range=[0, 105],
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig_var, use_container_width=True)

    st.markdown("**Canonical Variate Relationships**")

    n_plot = min(2, n_components)

    if n_plot >= 1:
        cols = st.columns(n_plot)
        for i in range(n_plot):
            with cols[i]:
                fig_scatter = go.Figure()
                fig_scatter.add_trace(
                    go.Scatter(
                        x=x_c[:, i],
                        y=z_c[:, i],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=np.arange(n_samples),
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Time"),
                        ),
                        text=[f"t={t}" for t in range(n_samples)],
                        hovertemplate=f"<b>Time:</b> %{{text}}<br>{x_label}: %{{x:.3f}}<br>{z_label}: %{{y:.3f}}<extra></extra>",
                    )
                )

                min_val = min(x_c[:, i].min(), z_c[:, i].min())
                max_val = max(x_c[:, i].max(), z_c[:, i].max())
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        line=dict(color="red", dash="dash", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                fig_scatter.update_layout(
                    title=f"Component {i+1} (r={canonical_corrs[i]:.3f})",
                    xaxis_title=f"{x_label} CC{i+1}",
                    yaxis_title=f"{z_label} CC{i+1}",
                    height=400,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("**CCA Loadings (Weights)**")
    st.markdown(
        "Shows how much each original variable contributes to each canonical component"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{x_label} Loadings**")
        x_loadings = np.corrcoef(x_p.T, x_c.T)[:n_features_x, n_features_x:]

        fig_load_x = go.Figure(
            data=go.Heatmap(
                z=x_loadings,
                x=[f"CC{i+1}" for i in range(n_components)],
                y=[f"Dim {i}" for i in range(n_features_x)],
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="Loading"),
            )
        )
        fig_load_x.update_layout(
            title=f"{x_label} Loadings",
            xaxis_title="Canonical Component",
            yaxis_title="Original Dimension",
            height=300,
        )
        st.plotly_chart(fig_load_x, use_container_width=True)

    with col2:
        st.markdown(f"**{z_label} Loadings**")
        z_loadings = np.corrcoef(z_data.T, z_c.T)[:n_features_z, n_features_z:]

        fig_load_z = go.Figure(
            data=go.Heatmap(
                z=z_loadings,
                x=[f"CC{i+1}" for i in range(n_components)],
                y=[f"Dim {i}" for i in range(n_features_z)],
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="Loading"),
            )
        )
        fig_load_z.update_layout(
            title=f"{z_label} Loadings",
            xaxis_title="Canonical Component",
            yaxis_title="Original Dimension",
            height=300,
        )
        st.plotly_chart(fig_load_z, use_container_width=True)

    st.markdown("**Summary Statistics**")
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Components", n_components)
    with stat_cols[1]:
        st.metric("Max Correlation", f"{canonical_corrs[0]:.3f}")
    with stat_cols[2]:
        st.metric("Mean Correlation", f"{np.mean(canonical_corrs):.3f}")
    with stat_cols[3]:
        strong_corr = np.sum(canonical_corrs > 0.5)
        st.metric("Strong Corr (r>0.5)", f"{strong_corr}/{n_components}")


def render_eigenvalue_analysis(x_p: np.ndarray, trial_idx: int):
    if x_p.ndim != 2:
        st.warning("Latent states must be 2D for eigenvalue analysis.")
        return

    n_samples, n_dims = x_p.shape

    x_centered = x_p - np.mean(x_p, axis=0)
    cov_matrix = np.cov(x_centered.T)

    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real

    total_var = np.sum(eigenvalues)
    if total_var == 0:
        explained_var = np.zeros_like(eigenvalues)
    else:
        explained_var = eigenvalues / total_var * 100
    cumulative_var = np.cumsum(explained_var)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Eigenvalue Spectrum")
        fig_eigen = go.Figure()
        fig_eigen.add_trace(
            go.Bar(
                x=list(range(1, len(eigenvalues) + 1)),
                y=eigenvalues,
                marker_color="steelblue",
                text=[f"{ev:.2f}" for ev in eigenvalues],
                textposition="outside",
            )
        )
        fig_eigen.update_layout(
            title=f"Eigenvalues of Latent Covariance — Trial {trial_idx}",
            xaxis_title="Component",
            yaxis_title="Eigenvalue",
            height=400,
        )
        st.plotly_chart(fig_eigen, use_container_width=True)

    with col2:
        st.markdown("##### Explained Variance")
        fig_var = go.Figure()
        fig_var.add_trace(
            go.Bar(
                x=list(range(1, len(explained_var) + 1)),
                y=explained_var,
                name="Individual",
                marker_color="lightblue",
            )
        )
        fig_var.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative_var) + 1)),
                y=cumulative_var,
                name="Cumulative",
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(size=8),
            )
        )
        fig_var.update_layout(
            title=f"Variance Explained — Trial {trial_idx}",
            xaxis_title="Component",
            yaxis_title="Variance Explained (%)",
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig_var, use_container_width=True)

    st.markdown("##### Statistics")
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Total Dimensions", n_dims)
    with stat_cols[1]:
        st.metric("Top Eigenvalue", f"{eigenvalues[0]:.2f}")
    with stat_cols[2]:
        n_95 = (
            np.argmax(cumulative_var >= 95) + 1
            if np.any(cumulative_var >= 95)
            else n_dims
        )
        st.metric("Dims for 95% Var", n_95)
    with stat_cols[3]:
        effective_dim = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues**2)
        st.metric("Effective Dimensionality", f"{effective_dim:.1f}")


def render_phase_space_analysis(
    split_res: Dict[str, Any],
    trial_idx: int,
    x_p: np.ndarray,
    z_p: np.ndarray = None,
):

    if x_p.ndim == 1:
        st.info("Phase space analysis requires at least 2 latent dimensions")
        return

    if x_p.shape[0] < x_p.shape[1]:
        x_p = x_p.T

    n_dims = x_p.shape[1]

    if n_dims < 2:
        st.info("Phase space analysis requires at least 2 latent dimensions")
        return

    col1, col2 = st.columns(2)
    with col1:
        dim_x = st.selectbox(
            "X-axis dimension",
            options=list(range(n_dims)),
            format_func=lambda x: f"Dimension {x+1}",
            key=f"phase_dim_x_{trial_idx}",
        )
    with col2:
        dim_y = st.selectbox(
            "Y-axis dimension",
            options=list(range(n_dims)),
            index=min(1, n_dims - 1),
            format_func=lambda x: f"Dimension {x+1}",
            key=f"phase_dim_y_{trial_idx}",
        )

    if dim_x == dim_y:
        st.warning("Please select different dimensions for each axis")
        return

    st.markdown("#### Single Trial Heatmap")
    _render_single_trial_heatmap(x_p, dim_x, dim_y, z_p, trial_idx)

    st.markdown("---")
    st.markdown("#### DBS ON/OFF Comparison")
    _render_dbs_comparison(split_res, dim_x, dim_y, trial_idx)


def _render_single_trial_heatmap(
    x_p: np.ndarray, dim_x: int, dim_y: int, z_p: np.ndarray, trial_idx: int
):

    x_data = x_p[:, dim_x]
    y_data = x_p[:, dim_y]

    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]

    if len(x_clean) < 10:
        st.warning("Not enough valid data points for heatmap")
        return

    n_bins = 50
    hist, x_edges, y_edges = np.histogram2d(x_clean, y_clean, bins=n_bins)
    hist_smooth = gaussian_filter(hist.T, sigma=1.5)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=x_edges[:-1],
            y=y_edges[:-1],
            z=hist_smooth,
            colorscale="Viridis",
            colorbar=dict(title="Density"),
            hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Density: %{z:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Phase Space Heatmap: Dimension {dim_x+1} vs Dimension {dim_y+1}",
        xaxis_title=f"Latent Dimension {dim_x+1}",
        yaxis_title=f"Latent Dimension {dim_y+1}",
        showlegend=False,
        hovermode="closest",
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_dbs_comparison(
    split_res: Dict[str, Any], dim_x: int, dim_y: int, trial_idx: int
):

    Xp_list = split_res.get("Xp", [])
    stim_list = split_res.get("stim", [])

    if not Xp_list or not stim_list:
        st.warning("DBS stimulation data not available")
        return

    # Separate trials by DBS condition
    dbs_on_trials = []
    dbs_off_trials = []

    for idx, stim in enumerate(stim_list):
        if idx >= len(Xp_list):
            continue
        if stim == "on":
            dbs_on_trials.append(idx)
        elif stim == "off":
            dbs_off_trials.append(idx)

    if not dbs_on_trials and not dbs_off_trials:
        st.warning("No DBS trials available for comparison")
        return

    if not dbs_on_trials:
        st.info(f"Only DBS OFF trials available ({len(dbs_off_trials)} trials)")
    elif not dbs_off_trials:
        st.info(f"Only DBS ON trials available ({len(dbs_on_trials)} trials)")

    def aggregate_phase_space(trial_indices):
        all_x = []
        all_y = []
        for idx in trial_indices:
            x_trial = np.array(Xp_list[idx])
            if x_trial.shape[0] < x_trial.shape[1]:
                x_trial = x_trial.T
            if x_trial.shape[1] <= max(dim_x, dim_y):
                continue
            x_data = x_trial[:, dim_x]
            y_data = x_trial[:, dim_y]
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
            all_x.extend(x_data[valid_mask])
            all_y.extend(y_data[valid_mask])
        return np.array(all_x), np.array(all_y)

    x_on, y_on = (
        aggregate_phase_space(dbs_on_trials)
        if dbs_on_trials
        else (np.array([]), np.array([]))
    )
    x_off, y_off = (
        aggregate_phase_space(dbs_off_trials)
        if dbs_off_trials
        else (np.array([]), np.array([]))
    )

    has_on = len(x_on) >= 10
    has_off = len(x_off) >= 10

    if not has_on and not has_off:
        st.warning("Not enough data points for visualization")
        return

    n_bins = 50

    all_x = []
    all_y = []
    if has_on:
        all_x.extend(x_on)
        all_y.extend(y_on)
    if has_off:
        all_x.extend(x_off)
        all_y.extend(y_off)

    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)

    x_bins = np.linspace(x_min, x_max, n_bins)
    y_bins = np.linspace(y_min, y_max, n_bins)

    hist_on_smooth = None
    hist_off_smooth = None

    if has_on:
        hist_on, _, _ = np.histogram2d(x_on, y_on, bins=[x_bins, y_bins])
        hist_on_smooth = gaussian_filter(hist_on.T, sigma=1.5)

    if has_off:
        hist_off, _, _ = np.histogram2d(x_off, y_off, bins=[x_bins, y_bins])
        hist_off_smooth = gaussian_filter(hist_off.T, sigma=1.5)

    from plotly.subplots import make_subplots

    if has_on and has_off:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"DBS ON ({len(dbs_on_trials)} trials)",
                f"DBS OFF ({len(dbs_off_trials)} trials)",
            ),
            horizontal_spacing=0.12,
        )

        fig.add_trace(
            go.Heatmap(
                x=x_bins[:-1],
                y=y_bins[:-1],
                z=hist_on_smooth,
                colorscale="Viridis",
                colorbar=dict(title="Density", x=0.45),
                hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Density: %{z:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=x_bins[:-1],
                y=y_bins[:-1],
                z=hist_off_smooth,
                colorscale="Viridis",
                colorbar=dict(title="Density", x=1.02),
                hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Density: %{z:.1f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text=f"Latent Dimension {dim_x+1}", row=1, col=1)
        fig.update_xaxes(title_text=f"Latent Dimension {dim_x+1}", row=1, col=2)
        fig.update_yaxes(title_text=f"Latent Dimension {dim_y+1}", row=1, col=1)
        fig.update_yaxes(title_text=f"Latent Dimension {dim_y+1}", row=1, col=2)

        fig.update_layout(
            title_text=f"DBS ON/OFF Phase Space Comparison: Dim {dim_x+1} vs Dim {dim_y+1}",
            height=600,
        )
    else:
        condition = "on" if has_on else "off"
        n_trials = len(dbs_on_trials) if has_on else len(dbs_off_trials)
        hist_smooth = hist_on_smooth if has_on else hist_off_smooth

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                x=x_bins[:-1],
                y=y_bins[:-1],
                z=hist_smooth,
                colorscale="Viridis",
                colorbar=dict(title="Density"),
                hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Density: %{z:.1f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"DBS {condition} Phase Space: Dim {dim_x+1} vs Dim {dim_y+1} ({n_trials} trials)",
            xaxis_title=f"Latent Dimension {dim_x+1}",
            yaxis_title=f"Latent Dimension {dim_y+1}",
            height=600,
        )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Distribution Statistics")

    if has_on and has_off:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**DBS ON**")
            st.write(f"Mean X: {x_on.mean():.4f} ± {x_on.std():.4f}")
            st.write(f"Mean Y: {y_on.mean():.4f} ± {y_on.std():.4f}")
        with col2:
            st.markdown("**DBS OFF**")
            st.write(f"Mean X: {x_off.mean():.4f} ± {x_off.std():.4f}")
            st.write(f"Mean Y: {y_off.mean():.4f} ± {y_off.std():.4f}")
    elif has_on:
        st.markdown("**DBS ON**")
        st.write(f"Mean X: {x_on.mean():.4f} ± {x_on.std():.4f}")
        st.write(f"Mean Y: {y_on.mean():.4f} ± {y_on.std():.4f}")
    else:
        st.markdown("**DBS OFF**")
        st.write(f"Mean X: {x_off.mean():.4f} ± {x_off.std():.4f}")
        st.write(f"Mean Y: {y_off.mean():.4f} ± {y_off.std():.4f}")


def render_latent_frequency_analysis(
    x_p: np.ndarray,
    y_true: np.ndarray,
    trial_idx: int,
    sampling_freq: float = 1000.0,
):

    st.markdown("### Frequency Analysis of Latent Dynamics")
    st.markdown("Compare frequency content of latent states vs neural signals")

    if x_p.ndim == 1:
        x_p = x_p.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    if x_p.shape[0] < x_p.shape[1]:
        x_p = x_p.T
    if y_true.shape[0] < y_true.shape[1]:
        y_true = y_true.T

    n_latent_dims = x_p.shape[1]
    n_neural_chans = y_true.shape[1]

    col1, col2 = st.columns(2)
    with col1:
        latent_dim = st.selectbox(
            "Latent dimension",
            options=list(range(n_latent_dims)),
            format_func=lambda x: f"Dimension {x+1}",
            key=f"latent_freq_dim_{trial_idx}",
        )
    with col2:
        neural_chan = st.selectbox(
            "Neural channel",
            options=list(range(n_neural_chans)),
            format_func=lambda x: f"Channel {x+1}",
            key=f"latent_freq_chan_{trial_idx}",
        )

    x_signal = x_p[:, latent_dim]
    y_signal = y_true[:, neural_chan]

    min_len = min(len(x_signal), len(y_signal))
    x_signal = x_signal[:min_len]
    y_signal = y_signal[:min_len]

    freqs_x, psd_x = compute_power_spectrum(x_signal, sampling_freq)
    freqs_y, psd_y = compute_power_spectrum(y_signal, sampling_freq)

    dom_freqs_x, dom_powers_x = find_dominant_frequencies(freqs_x, psd_x, n_peaks=5)
    dom_freqs_y, dom_powers_y = find_dominant_frequencies(freqs_y, psd_y, n_peaks=5)

    spec_corr = spectral_correlation(freqs_x, psd_x, psd_y)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=freqs_x,
            y=psd_x,
            mode="lines",
            name=f"Latent Dim {latent_dim+1}",
            line=dict(color="purple", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=freqs_y,
            y=psd_y,
            mode="lines",
            name=f"Neural Chan {neural_chan+1}",
            line=dict(color="orange", width=2, dash="dash"),
        )
    )

    if len(dom_freqs_x) > 0:
        fig.add_trace(
            go.Scatter(
                x=dom_freqs_x,
                y=dom_powers_x,
                mode="markers",
                name="Latent Peaks",
                marker=dict(size=10, color="purple", symbol="x"),
            )
        )

    if len(dom_freqs_y) > 0:
        fig.add_trace(
            go.Scatter(
                x=dom_freqs_y,
                y=dom_powers_y,
                mode="markers",
                name="Neural Peaks",
                marker=dict(size=10, color="orange", symbol="circle"),
            )
        )

    fig.update_layout(
        title=f"Power Spectrum: Latent Dim {latent_dim+1} vs Neural Chan {neural_chan+1}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density",
        yaxis_type="log",
        showlegend=True,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### Dominant Frequencies (Latent Dim {latent_dim+1})")
        if len(dom_freqs_x) > 0:
            for i, (freq, power) in enumerate(zip(dom_freqs_x, dom_powers_x)):
                st.metric(f"Peak {i+1}", f"{freq:.2f} Hz", f"Power: {power:.2e}")
        else:
            st.info("No dominant peaks detected")

    with col2:
        st.markdown(f"#### Dominant Frequencies (Neural Chan {neural_chan+1})")
        if len(dom_freqs_y) > 0:
            for i, (freq, power) in enumerate(zip(dom_freqs_y, dom_powers_y)):
                st.metric(f"Peak {i+1}", f"{freq:.2f} Hz", f"Power: {power:.2e}")
        else:
            st.info("No dominant peaks detected")

    st.markdown("#### Spectral Correlation")
    st.metric(
        "Correlation between spectra",
        f"{spec_corr:.4f}",
    )


def render_latent_states_tab(split_res: Dict[str, Any], trial_idx: int):
    Xp = split_res["Xp"]
    Zp = split_res["Zp"]

    x_p = np.array(Xp[trial_idx])
    z_p = None if Zp[trial_idx] is None else np.array(Zp[trial_idx])

    Z_true = split_res.get("Z", [])
    z_true = None
    if Z_true and len(Z_true) > trial_idx and Z_true[trial_idx] is not None:
        z_true = np.array(Z_true[trial_idx])

    offsets = split_res.get("offset", [])
    t_offset = (
        float(offsets[trial_idx])
        if offsets and len(offsets) > trial_idx and offsets[trial_idx] is not None
        else 0.0
    )

    Y_true = split_res["Y"]
    y_t = np.array(Y_true[trial_idx])
    n_samples = y_t.shape[0]
    t_abs = get_trial_time_axis(split_res, trial_idx, n_samples, t_offset)

    cm_list = split_res.get("chunk_margin", [])
    md_list = split_res.get("margined_duration", [])
    cm = cm_list[trial_idx] if cm_list else None
    dur = md_list[trial_idx] if md_list else None

    if x_p is not None:
        st.markdown("### Time Series")
        st.markdown("Latent state trajectories over time")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Latent States (X_p)")
            render_latent_states_plot(x_p, t_abs, t_offset, cm, dur, trial_idx)

        with col2:
            if z_p is not None:
                st.markdown("#### Other Predictions (Z_p)")
                render_auxiliary_predictions_plot(
                    z_p, t_abs, t_offset, cm, dur, trial_idx
                )

        st.markdown("---")
        st.markdown("### Dimensionality Reduction")
        st.markdown("Low-dimensional projections of latent space")
        render_tsne_umap_plot(x_p, trial_idx, z_p=z_true)

        st.markdown("---")
        st.markdown("### Eigenvalue Analysis")
        render_eigenvalue_analysis(x_p, trial_idx)

        st.markdown("---")
        st.markdown("### Phase Space Analysis")
        st.markdown("Comprehensive phase space visualization with multiple views")
        render_phase_space_analysis(split_res, trial_idx, x_p, z_p=z_true)

        st.markdown("---")
        cfg_path = st.session_state.get("config_path")
        cfg = get_config(str(cfg_path))
        sampling_freq = cfg.data.sampling_frequency

        render_latent_frequency_analysis(
            x_p,
            y_t,
            trial_idx,
            sampling_freq,
        )

        st.markdown("---")
        st.markdown("### Canonical Correlation Analysis")
        st.markdown("Analyze relationships between latent states and other variables")

        cca_options = []
        if z_true is not None and z_true.ndim == 2:
            cca_options.append("Latent States (X) vs Behavioral Variables (Z)")
        if y_t is not None and y_t.ndim == 2:
            cca_options.append("Latent States (X) vs Neural Signals (Y)")

        if len(cca_options) > 0:
            cca_choice = st.selectbox(
                "Select CCA analysis",
                options=cca_options,
                key=f"cca_choice_{trial_idx}",
            )

            if cca_choice == "Latent States (X) vs Behavioral Variables (Z)":
                render_cca_analysis(
                    x_p,
                    z_true,
                    trial_idx,
                    x_label="Latent States (X)",
                    z_label="Behavioral Variables (Z)",
                )
            elif cca_choice == "Latent States (X) vs Neural Signals (Y)":
                render_cca_analysis(
                    x_p,
                    y_t,
                    trial_idx,
                    x_label="Latent States (X)",
                    z_label="Neural Signals (Y)",
                )
        else:
            st.info(
                "CCA analysis requires at least one additional variable (Z or Y) with multiple dimensions."
            )

    else:
        st.info("No latent states available for this trial.")


def model_predictions_tab(project_root):
    st.header("Model Predictions")

    RESULTS_ROOT = project_root / "results"

    variants = list_variants(RESULTS_ROOT)
    if len(variants) == 0:
        st.info("No result variants found under results/.")
        return

    variant = st.selectbox("Model variant", options=variants, key="pred_variant")
    variant_dir = RESULTS_ROOT / variant
    runs = list_run_timestamps(variant_dir)

    if len(runs) == 0:
        st.info("No runs found for this variant yet. Train a model first.")
        return

    run_ts = st.selectbox("Run timestamp", options=runs, key="pred_run")
    cfg_path = config_for_variant(project_root, variant)

    if cfg_path is None:
        st.error(
            f"Config not found for variant '{variant}'. Expected at training/setups/{variant}.yaml"
        )
        return

    available_results = check_precomputed_results(variant_dir, run_ts)

    st.markdown("### Select Splits to Visualize")
    selected_splits = st.multiselect(
        "Choose splits",
        options=["train", "val", "test"],
        default=["val"] if available_results.get("val") else [],
        key="selected_splits",
    )

    button_label = "Load/Compute Results" if selected_splits else "Select splits first"
    if st.button(button_label, key="btn_load_results", disabled=not selected_splits):
        st.session_state["predictions_key"] = (
            str(cfg_path),
            run_ts,
            tuple(selected_splits),
        )

    @st.cache_resource(show_spinner=True)
    def _cached_predictions_selective(
        config_path: str, run_timestamp: str, splits_to_compute: tuple
    ):
        return compute_predictions_selective(
            config_path, run_timestamp, splits_to_compute
        )

    pred_key = st.session_state.get("predictions_key")
    if pred_key and pred_key[0] == str(cfg_path) and pred_key[1] == run_ts:
        requested_splits = pred_key[2] if len(pred_key) > 2 else selected_splits

        pred_results = {}

        with st.spinner("Loading/computing results..."):
            for split in requested_splits:
                if available_results.get(split):
                    with st.spinner(f"Loading pre-computed {split} results..."):
                        loaded = load_precomputed_results(variant_dir, run_ts, split)
                        if loaded:
                            pred_results[split] = loaded
                        else:
                            st.error(f"Failed to load {split} results")

            from utils.config import get_config

            cfg = get_config(str(cfg_path))
            input_chans = cfg.data.channels.input
            output_chans = cfg.data.channels.output

            for split, res in pred_results.items():
                if not res.get("input_channels") and input_chans:
                    res["input_channels"] = input_chans
                if not res.get("output_channels") and output_chans:
                    res["output_channels"] = output_chans

            splits_to_compute = tuple(
                [s for s in requested_splits if not available_results.get(s)]
            )
            if splits_to_compute:
                with st.spinner(f"Computing {', '.join(splits_to_compute)} results..."):
                    try:
                        computed = _cached_predictions_selective(
                            str(cfg_path), run_ts, splits_to_compute
                        )
                        pred_results.update(computed)
                    except Exception as e:
                        st.error(f"Computation failed: {e}")

        if pred_results:
            split = st.selectbox(
                "Split to visualize",
                options=list(pred_results.keys()),
                key="pred_split",
            )
            split_res = pred_results.get(split)

            if not split_res:
                st.info("No results for selected split.")
                return

            Y_true = split_res["Y"]
            n_trials = len(Y_true)
            trial_indices = list(range(n_trials))
            trial_idx = st.selectbox("Trial", options=trial_indices, key="pred_trial")

            pid_list = split_res.get("participant_id", [])
            ses_list = split_res.get("session", [])
            blk_list = split_res.get("block", [])
            tri_list = split_res.get("trial", [])

            hdr_pid = (
                pid_list[trial_idx]
                if pid_list
                else st.session_state.get("participant_id")
            )
            hdr_ses = (
                ses_list[trial_idx] if ses_list else st.session_state.get("session")
            )
            hdr_blk = blk_list[trial_idx] if blk_list else st.session_state.get("block")
            hdr_tri = tri_list[trial_idx] if tri_list else trial_idx

            st.subheader(
                f"Participant {hdr_pid} | Session {hdr_ses} | Block {hdr_blk} | Trial {hdr_tri}"
            )

            predictions_subtab, forecasting_subtab, latent_states_subtab = st.tabs(
                ["Predictions", "Forecasting", "Latent States"]
            )

            st.session_state["config_path"] = cfg_path
            st.session_state["run_timestamp"] = run_ts

            with predictions_subtab:
                render_predictions_tab(split_res, trial_idx, cfg_path)

            with forecasting_subtab:
                render_forecasting_tab(
                    split_res, trial_idx, cfg_path, run_ts, Y_true, split_res["Yp"]
                )

            with latent_states_subtab:
                render_latent_states_tab(split_res, trial_idx)
