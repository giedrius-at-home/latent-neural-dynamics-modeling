import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any
from scipy import stats as scipy_stats

from training.components.tester import Tester
from dashboard.backbone import PALETTE
from dashboard.subtabs.helpers import get_trial_time_axis, transpose_if_needed
from utils.stats import (
    compute_residual_statistics,
    qq_plot_data,
    normality_tests,
    probability_plot_data,
    whiteness_test,
)


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


def render_z_scatter_plot(
    z_true_c: np.ndarray,
    z_pred_c: np.ndarray,
    channel_name: str,
    r_ch: float,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=z_true_c,
            y=z_pred_c,
            mode="markers",
            name="Predictions",
            marker=dict(
                size=4,
                color="rgba(31, 119, 180, 0.6)",
                line=dict(width=0),
            ),
        )
    )

    min_val = min(np.min(z_true_c), np.min(z_pred_c))
    max_val = max(np.max(z_true_c), np.max(z_pred_c))
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
        z_true_c, z_pred_c
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
        xaxis_title="Z_true",
        yaxis_title="Z_pred",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### OLS Regression Coefficients")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Slope", f"{slope:.4f}")
    with col2:
        st.metric("Intercept", f"{intercept:.4f}")
    with col3:
        st.metric("R²", f"{r_squared:.4f}")
    with col4:
        st.metric("p-value", f"{p_value:.2e}")


def render_z_residual_plot(
    z_true_c: np.ndarray,
    z_pred_c: np.ndarray,
    t_abs: np.ndarray,
    channel_name: str,
):
    residuals = z_pred_c - z_true_c

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
        title=f"Residuals (Prediction Errors) — {channel_name} (RMSE={rmse:.3f})",
        xaxis_title="Time (s)",
        yaxis_title="Error",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_z_statistics_table(
    z_true_c: np.ndarray,
    z_pred_c: np.ndarray,
    r_ch: float,
    channel_name: str,
):
    residuals = z_pred_c - z_true_c
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((z_true_c - np.mean(z_true_c)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    st.markdown(f"### Statistics — {channel_name}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Pearson r", f"{r_ch:.4f}")

    with col2:
        st.metric("R²", f"{r_squared:.4f}")

    with col3:
        st.metric("RMSE", f"{rmse:.3f}")

    with col4:
        st.metric("MAE", f"{mae:.3f}")


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

            z_t_transposed = transpose_if_needed(z_t, len(t_abs))
            z_p_transposed = transpose_if_needed(z_p, len(t_abs))
            z_true_c = (
                z_t_transposed.squeeze() if nz_chan == 1 else z_t_transposed[:, z_c]
            )
            z_pred_c = (
                z_p_transposed.squeeze() if nz_chan == 1 else z_p_transposed[:, z_c]
            )

            st.markdown("#### Time Series: Z_true vs Z_pred")
            render_z_prediction_plot(z_t, z_p, t_abs, z_c, selected_z_name, r_z_ch)

            st.markdown("#### Scatter Plot: True vs Predicted")
            render_z_scatter_plot(z_true_c, z_pred_c, selected_z_name, r_z_ch)

            st.markdown("#### Residual Plot: Prediction Errors Over Time")
            render_z_residual_plot(z_true_c, z_pred_c, t_abs, selected_z_name)

            render_z_statistics_table(z_true_c, z_pred_c, r_z_ch, selected_z_name)

            st.markdown("---")
            with st.expander("Residual Diagnostics & Normality Tests", expanded=False):
                st.markdown(
                    "Comprehensive diagnostics to verify that residuals follow a Gaussian distribution."
                )
                render_residual_diagnostics(z_true_c, z_pred_c, selected_z_name)

            st.markdown("---")
            with st.expander("Detrended Z_pred Diagnostics", expanded=False):
                st.markdown(
                    "Analysis of Z_pred with mean removed to check if it represents white Gaussian noise."
                )
                z_pred_detrended = z_pred_c - np.mean(z_pred_c)
                z_true_zero = np.zeros_like(z_pred_detrended)
                render_residual_diagnostics(
                    z_true_zero, z_pred_detrended, f"{selected_z_name} (detrended)"
                )
