import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, Optional
from scipy import stats as scipy_stats
import pandas as pd
import json

from training.components.tester import Tester
from dashboard.backbone import (
    PALETTE,
    PLOT_STYLE,
    PLOT_COLOR,
    create_base_time_series_figure,
    add_margin_visualization,
    add_caption_below,
)
from dashboard.subtabs.helpers import get_trial_time_axis, transpose_if_needed, rescale_to_reference

BASELINE_COLOR = "#00E5FF"
from utils.stats import (
    compute_residual_statistics,
    qq_plot_data,
    normality_tests,
    probability_plot_data,
    probability_plot_data,
    whiteness_test,
    compute_power_spectrum,
    find_dominant_frequencies,
    spectral_correlation,
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

    onset_time = t_abs.min() if len(t_abs) > 0 else 0.0
    fig = create_base_time_series_figure(
        time_abs=t_abs,
        onset_time=onset_time,
        y_label="Amplitude (µV)",
        title="",
    )

    fig.add_trace(
        go.Scatter(
            x=t_abs,
            y=y_true_c,
            name="True",
            mode="lines",
            line=dict(color=PLOT_COLOR.stim_off, width=PLOT_STYLE.line_width_normal),
        )
    )
    if y_pred_c is not None:
        fig.add_trace(
            go.Scatter(
                x=t_abs,
                y=y_pred_c,
                name="Predicted",
                mode="lines",
                line=dict(
                    color=PLOT_COLOR.stim_on,
                    width=PLOT_STYLE.line_width_normal,
                    dash="dot",
                ),
            )
        )

    st.plotly_chart(fig, use_container_width=True)


def render_y_scatter_plot(
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    channel_name: str,
    r_ch: float,
    baseline_preds: Optional[np.ndarray] = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_true_c,
            y=y_pred_c,
            mode="markers",
            name=model_name,
            marker=dict(
                size=4,
                color=PALETTE.twilight_indigo,
                opacity=0.6,
                line=dict(width=0),
            ),
        )
    )

    if baseline_preds is not None:
        fig.add_trace(
            go.Scatter(
                x=y_true_c,
                y=baseline_preds,
                mode="markers",
                name=baseline_name,
                marker=dict(
                    size=4,
                    color=BASELINE_COLOR,
                    opacity=0.4,
                    line=dict(width=0),
                ),
            )
        )

    min_val = min(np.min(y_true_c), np.min(y_pred_c))
    max_val = max(np.max(y_true_c), np.max(y_pred_c))

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
            name=f"OLS ({model_name})",
            line=dict(color=PALETTE.strawberry_red, width=1.2),
        )
    )

    if baseline_preds is not None:
        slope_b, intercept_b, r_value_b, p_value_b, _ = scipy_stats.linregress(
            y_true_c, baseline_preds
        )
        y_line_b = slope_b * x_line + intercept_b
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line_b,
                mode="lines",
                name=f"OLS ({baseline_name})",
                line=dict(color=BASELINE_COLOR, width=1.2, dash="dash"),
            )
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        margin=dict(l=60, r=80, t=40, b=60),
        xaxis_title="True Amplitude (µV)",
        yaxis_title="Predicted Amplitude (µV)",
    )
    st.plotly_chart(fig, use_container_width=True)

    caption_parts = [f"True vs Predicted — {channel_name} (Pearson r={r_ch:.3f})"]
    caption_parts.append(
        f"{model_name} OLS: y={slope:.4f}x+{intercept:.4f}, R²={r_squared:.4f}, p={p_value:.2e}"
    )
    if baseline_preds is not None:
        r_sq_b = r_value_b ** 2
        caption_parts.append(
            f"{baseline_name} OLS: y={slope_b:.4f}x+{intercept_b:.4f}, R²={r_sq_b:.4f}, p={p_value_b:.2e}"
        )
    st.caption(" | ".join(caption_parts))


def render_y_residual_plot(
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    t_abs: np.ndarray,
    channel_name: str,
    chunk_margin: float = 0.0,
    baseline_preds: Optional[np.ndarray] = None,
    baseline_name: str = "Baseline",
):
    residuals = y_true_c - y_pred_c

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
        baseline_caption = f" | {baseline_name} RMSE={base_rmse:.3f} µV"

    if chunk_margin > 0:
        add_margin_visualization(fig, t_abs, chunk_margin)

    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        margin=dict(l=60, r=80, t=40, b=60),
        xaxis_title="Time (s)",
        yaxis_title="Residual (µV)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Residuals (Prediction Errors) — {channel_name} (RMSE={rmse:.3f} µV){baseline_caption}")


def render_statistics_table(
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    r_ch: float,
    channel_name: str,
):
    residuals = y_true_c - y_pred_c
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
        st.metric("R2", f"{r_squared:.4f}")

    with col3:
        st.metric("RMSE (µV)", f"{rmse:.3f}")

    with col4:
        st.metric("MAE (µV)", f"{mae:.3f}")


def _show_residual_stats_row(stats: dict, label: str = ""):
    """Display a row of residual stats metrics."""
    if label:
        st.markdown(f"**{label}**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Mean", f"{stats['mean']:.4f}")
    with c2:
        st.metric("Std Dev", f"{stats['std']:.4f}")
    with c3:
        st.metric("Min", f"{stats['min']:.4f}")
    with c4:
        st.metric("Max", f"{stats['max']:.4f}")


def render_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    channel_name: str,
    baseline_preds: Optional[np.ndarray] = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
):

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

    norm_tests = normality_tests(residuals)

    st.markdown("#### Normality Test")
    shapiro_stat, shapiro_p = norm_tests["shapiro"]
    if baseline_preds is not None:
        base_norm = normality_tests(base_residuals)
        base_shapiro_stat, base_shapiro_p = base_norm["shapiro"]
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric(
                f"Shapiro-Wilk ({model_name})",
                f"p = {shapiro_p:.4f}",
                delta=f"stat = {shapiro_stat:.4f}",
            )
        with col_s2:
            st.metric(
                f"Shapiro-Wilk ({baseline_name})",
                f"p = {base_shapiro_p:.4f}",
                delta=f"stat = {base_shapiro_stat:.4f}",
            )
    else:
        st.metric(
            "Shapiro-Wilk Test",
            f"p = {shapiro_p:.4f}",
            delta=f"stat = {shapiro_stat:.4f}",
        )

    st.markdown("#### Q-Q Plot (Quantile-Quantile)")

    residuals_std = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)
    theoretical_q, sample_q = qq_plot_data(residuals_std)

    if len(theoretical_q) > 0:
        fig_qq = go.Figure()

        fig_qq.add_trace(
            go.Scatter(
                x=theoretical_q,
                y=sample_q,
                mode="markers",
                name=model_name,
                marker=dict(size=4, color=PALETTE.twilight_indigo, opacity=0.6),
            )
        )

        if baseline_preds is not None:
            base_resids = y_true - baseline_preds
            base_resids_std = (base_resids - np.mean(base_resids)) / (np.std(base_resids) + 1e-12)
            base_tq, base_sq = qq_plot_data(base_resids_std)
            if len(base_tq) > 0:
                fig_qq.add_trace(
                    go.Scatter(
                        x=base_tq,
                        y=base_sq,
                        mode="markers",
                        name=baseline_name,
                        marker=dict(size=4, color=BASELINE_COLOR, opacity=0.4),
                    )
                )

        min_q = min(np.min(theoretical_q), np.min(sample_q))
        max_q = max(np.max(theoretical_q), np.max(sample_q))
        fig_qq.add_trace(
            go.Scatter(
                x=[min_q, max_q],
                y=[min_q, max_q],
                mode="lines",
                name="Normal",
                line=dict(color=PALETTE.strawberry_red, width=1, dash="dash"),
            )
        )

        fig_qq.update_layout(
            template="plotly_white",
            font=dict(family=PLOT_STYLE.font_family),
            margin=dict(l=60, r=80, t=40, b=60),
            showlegend=True,
            xaxis_title="Theoretical Quantiles (Normal)",
            yaxis_title="Sample Quantiles (Standardized)",
        )
        st.plotly_chart(fig_qq, use_container_width=True)
        st.caption(f"Q-Q Plot — {channel_name} (Standardized)")
    else:
        pass

    st.markdown("#### Probability Plot (CDF Comparison)")

    theoretical_cdf, empirical_cdf = probability_plot_data(residuals_std)

    if len(theoretical_cdf) > 0:
        fig_prob = go.Figure()

        fig_prob.add_trace(
            go.Scatter(
                x=theoretical_cdf,
                y=empirical_cdf,
                mode="markers",
                name=model_name,
                marker=dict(size=3, color=PALETTE.twilight_indigo),
            )
        )

        if baseline_preds is not None:
            base_resids = y_true - baseline_preds
            base_resids_std = (base_resids - np.mean(base_resids)) / (np.std(base_resids) + 1e-12)
            base_tcdf, base_ecdf = probability_plot_data(base_resids_std)
            if len(base_tcdf) > 0:
                fig_prob.add_trace(
                    go.Scatter(
                        x=base_tcdf,
                        y=base_ecdf,
                        mode="markers",
                        name=baseline_name,
                        marker=dict(size=3, color=BASELINE_COLOR, opacity=0.5),
                    )
                )

        fig_prob.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Reference",
                line=dict(color=PALETTE.strawberry_red, width=1, dash="dash"),
            )
        )

        fig_prob.update_layout(
            template="plotly_white",
            font=dict(family=PLOT_STYLE.font_family),
            margin=dict(l=60, r=80, t=40, b=60),
            showlegend=True,
            xaxis_title="Theoretical Cumulative Probability",
            yaxis_title="Empirical Cumulative Probability",
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        st.caption(f"Probability Plot — {channel_name}")
    else:
        pass

    st.markdown("#### Whiteness Test (Ljung-Box)")
    whiteness_results = whiteness_test(residuals)
    lb_stat = whiteness_results["ljung_box_stat"]
    lb_p = whiteness_results["ljung_box_p"]
    lags = whiteness_results["lags"]
    acf = whiteness_results["acf"]

    if baseline_preds is not None:
        base_resids_w = y_true - baseline_preds
        base_whiteness = whiteness_test(base_resids_w)
        base_lb_stat = base_whiteness["ljung_box_stat"]
        base_lb_p = base_whiteness["ljung_box_p"]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                f"LB Stat ({model_name})",
                f"{lb_stat:.4f}" if not np.isnan(lb_stat) else "N/A",
            )
        with col2:
            st.metric(
                f"LB p ({model_name})",
                f"{lb_p:.4f}" if not np.isnan(lb_p) else "N/A",
            )
        with col3:
            st.metric(
                f"LB Stat ({baseline_name})",
                f"{base_lb_stat:.4f}" if not np.isnan(base_lb_stat) else "N/A",
            )
        with col4:
            st.metric(
                f"LB p ({baseline_name})",
                f"{base_lb_p:.4f}" if not np.isnan(base_lb_p) else "N/A",
            )
    else:
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
                name=model_name,
                marker=dict(color=PALETTE.twilight_indigo),
            )
        )

        if baseline_preds is not None:
            base_acf_vals = base_whiteness["acf"]
            base_lags = base_whiteness["lags"]
            if len(base_lags) > 0:
                fig_acf.add_trace(
                    go.Bar(
                        x=base_lags,
                        y=base_acf_vals,
                        name=baseline_name,
                        marker=dict(color=BASELINE_COLOR, opacity=0.5),
                    )
                )

        confidence_interval = 1.96 / np.sqrt(len(residuals.flatten()))
        fig_acf.add_hline(
            y=confidence_interval,
            line_dash="dot",
            line_color=PALETTE.cool_steel,
            annotation_text="95% CI",
        )
        fig_acf.add_hline(
            y=-confidence_interval,
            line_dash="dot",
            line_color=PALETTE.cool_steel,
        )

        fig_acf.update_layout(
            template="plotly_white",
            font=dict(family=PLOT_STYLE.font_family),
            margin=dict(l=60, r=80, t=40, b=60),
            showlegend=True,
            barmode="group",
            xaxis_title="Lag (timesteps)",
            yaxis_title="Autocorrelation",
        )
        st.plotly_chart(fig_acf, use_container_width=True)
        st.caption(f"Autocorrelation Function — {channel_name}")
    else:
        pass


def render_system_eigenvalues(idSys):
    A = getattr(idSys, "A", None)
    if A is None:
        return

    nx_brain = getattr(idSys, "nx_brain", A.shape[0])
    nx_noise = getattr(idSys, "nx_noise", 0)

    st.markdown("---")
    st.markdown("### System Dynamics Stability")

    A_np = np.array(A) if not isinstance(A, np.ndarray) else A
    if A_np.size == 0:
        st.info("System matrix A is empty")
        return

    A_brain = A_np[:nx_brain, :nx_brain]
    ev_brain = np.linalg.eigvals(A_brain) if nx_brain > 0 else np.array([])

    ev_noise = np.array([])
    if nx_noise > 0:
        A_noise = A_np[nx_brain:, nx_brain:]
        ev_noise = np.linalg.eigvals(A_noise)

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = go.Figure()

        # Unit circle for stability reference
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(
            go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode="lines",
                line=dict(color="rgba(150, 150, 150, 0.5)", dash="dash"),
                name="Unit Circle",
                showlegend=False,
            )
        )

        if ev_brain.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=ev_brain.real,
                    y=ev_brain.imag,
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=PLOT_COLOR.stim_on,
                        line=dict(width=1, color="black"),
                    ),
                    name="Dynamics",
                    hovertemplate="<b>Brain - Real:</b> %{x:.4f}<br><b>Imag:</b> %{y:.4f}<br><b>Magnitude:</b> %{customdata:.4f}<extra></extra>",
                    customdata=np.abs(ev_brain),
                )
            )

        if ev_noise.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=ev_noise.real,
                    y=ev_noise.imag,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=PALETTE.strawberry_red,
                        symbol="diamond",
                        line=dict(width=1, color="black"),
                    ),
                    name="Behavior Noise",
                    hovertemplate="<b>Noise - Real:</b> %{x:.4f}<br><b>Imag:</b> %{y:.4f}<br><b>Magnitude:</b> %{customdata:.4f}<extra></extra>",
                    customdata=np.abs(ev_noise),
                )
            )

        max_val = 1.1
        if ev_brain.size > 0 or ev_noise.size > 0:
            all_ev = np.concatenate([ev_brain, ev_noise])
            max_val = max(1.1, np.max(np.abs(all_ev)) * 1.1)

        fig.add_shape(
            type="line",
            x0=-max_val,
            y0=0,
            x1=max_val,
            y1=0,
            line=dict(color="lightgray", width=1),
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=-max_val,
            x1=0,
            y1=max_val,
            line=dict(color="lightgray", width=1),
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Real",
            yaxis_title="Imaginary",
            xaxis=dict(range=[-max_val, max_val], scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-max_val, max_val], scaleanchor="x", scaleratio=1),
            width=450,
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Eigenvalues of the augmented system matrix A. Points inside the unit circle indicate stable dynamics."
        )

    with col2:
        if ev_brain.size > 0:
            st.markdown("#### Eigenvalues")
            df_brain = pd.DataFrame(
                {
                    "Real": ev_brain.real,
                    "Imag": ev_brain.imag,
                    "Magnitude": np.abs(ev_brain),
                    "Phase (rad)": np.angle(ev_brain),
                }
            )
            st.dataframe(
                df_brain.style.format("{:.4f}"),
                hide_index=True,
                use_container_width=True,
            )
            max_eig_b = np.max(np.abs(ev_brain))
            st.info(
                f"**Spectral Radius:** {max_eig_b:.4f} ({'Stable' if max_eig_b < 1.0 else 'Unstable'})"
            )

        if ev_noise.size > 0:
            st.markdown("#### Behavior Noise Eigenvalues")
            df_noise = pd.DataFrame(
                {
                    "Real": ev_noise.real,
                    "Imag": ev_noise.imag,
                    "Magnitude": np.abs(ev_noise),
                    "Phase (rad)": np.angle(ev_noise),
                }
            )
            st.dataframe(
                df_noise.style.format("{:.4f}"),
                hide_index=True,
                use_container_width=True,
            )
            max_eig_n = np.max(np.abs(ev_noise))
            st.info(
                f"**Noise Spectral Radius:** {max_eig_n:.4f} ({'Stable' if max_eig_n < 1.0 else 'Unstable'})"
            )


def render_learned_noise_diagnostics(
    config_path: str,
    run_ts: str,
):
    st.markdown("### Learned Noise Covariance Matrices")
    with st.spinner("Loading model..."):
        tester = Tester.from_config_file(config_path, run_timestamp=run_ts)
        tester._load_model_for_run()

    model = tester.framework.model

    if hasattr(model, "idSys"):
        idSys = model.idSys

        # System matrix eigenvalues
        render_system_eigenvalues(idSys)

        st.markdown("---")
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
    baseline_preds: Optional[np.ndarray] = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
):
    z_pred_c_rescaled = rescale_to_reference(z_pred_c, z_true_c)
    baseline_rescaled = rescale_to_reference(baseline_preds, z_true_c) if baseline_preds is not None else None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=z_true_c,
            y=z_pred_c_rescaled,
            mode="markers",
            name=model_name,
            marker=dict(
                size=4,
                color=PALETTE.twilight_indigo,
                opacity=0.6,
                line=dict(width=0),
            ),
        )
    )

    if baseline_rescaled is not None:
        fig.add_trace(
            go.Scatter(
                x=z_true_c,
                y=baseline_rescaled,
                mode="markers",
                name=baseline_name,
                marker=dict(
                    size=4,
                    color=BASELINE_COLOR,
                    opacity=0.4,
                    line=dict(width=0),
                ),
            )
        )

    min_val = min(np.min(z_true_c), np.min(z_pred_c_rescaled))
    max_val = max(np.max(z_true_c), np.max(z_pred_c_rescaled))

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        z_true_c, z_pred_c_rescaled
    )
    r_squared = r_value**2

    x_line = np.array([min_val, max_val])
    y_line = slope * x_line + intercept

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"OLS ({model_name})",
            line=dict(color=PALETTE.strawberry_red, width=1.2),
        )
    )

    if baseline_rescaled is not None:
        slope_b, intercept_b, r_value_b, p_value_b, _ = scipy_stats.linregress(
            z_true_c, baseline_rescaled
        )
        y_line_b = slope_b * x_line + intercept_b
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line_b,
                mode="lines",
                name=f"OLS ({baseline_name})",
                line=dict(color=BASELINE_COLOR, width=1.2, dash="dash"),
            )
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        margin=dict(l=60, r=80, t=40, b=60),
        showlegend=True,
        xaxis_title="True Value",
        yaxis_title="Predicted Value (rescaled)",
    )
    st.plotly_chart(fig, use_container_width=True)

    caption_parts = [f"True vs Predicted — {channel_name} (Pearson r={r_ch:.3f})"]
    caption_parts.append(
        f"{model_name} OLS: y={slope:.4f}x+{intercept:.4f}, R²={r_squared:.4f}, p={p_value:.2e}"
    )
    if baseline_rescaled is not None:
        r_sq_b = r_value_b ** 2
        caption_parts.append(
            f"{baseline_name} OLS: y={slope_b:.4f}x+{intercept_b:.4f}, R²={r_sq_b:.4f}, p={p_value_b:.2e}"
        )
    st.caption(" | ".join(caption_parts))


def _compute_zp_components(z_true_c: np.ndarray, xp: np.ndarray, B_z: np.ndarray, d_z: np.ndarray, n1: int, channel_idx: int) -> tuple:
    if xp is None or B_z is None or n1 <= 0 or xp.shape[1] < n1:
        return None, None, np.nan, np.nan
        
    xp_1 = xp[:, :n1]
    xp_2 = xp[:, n1:]
    
    B_z_1 = B_z[channel_idx, :n1]
    B_z_2 = B_z[channel_idx, n1:]
    
    zp_1 = (xp_1 @ B_z_1)
    if d_z is not None and len(d_z) > channel_idx:
        zp_1 += d_z[channel_idx]
        
    zp_2 = (xp_2 @ B_z_2) 
    # Do not add d_z to zp_2 to avoid double counting the bias
    
    r_zp1 = np.corrcoef(z_true_c.flatten(), zp_1.flatten())[0, 1] if len(z_true_c) > 1 and len(zp_1) > 1 else np.nan
    r_zp2 = np.corrcoef(z_true_c.flatten(), zp_2.flatten())[0, 1] if len(z_true_c) > 1 and len(zp_2) > 1 else np.nan
    
    return zp_1, zp_2, r_zp1, r_zp2


def render_z_residual_plot(
    z_true_c: np.ndarray,
    z_pred_c: np.ndarray,
    t_abs: np.ndarray,
    channel_name: str,
    chunk_margin: float = 0.0,
    baseline_preds: Optional[np.ndarray] = None,
    baseline_name: str = "Baseline",
):
    residuals = z_true_c - z_pred_c

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
        baseline_residuals = z_true_c - baseline_preds
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
        baseline_caption = f" | {baseline_name} RMSE={base_rmse:.3f}"

    if chunk_margin > 0:
        add_margin_visualization(fig, t_abs, chunk_margin)

    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        margin=dict(l=60, r=80, t=40, b=60),
        xaxis_title="Time (s)",
        yaxis_title="Residual",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Residuals (Prediction Errors) — {channel_name} (RMSE={rmse:.3f}){baseline_caption}")


def render_z_statistics_table(
    z_true_c: np.ndarray,
    z_pred_c: np.ndarray,
    r_ch: float,
    channel_name: str,
    baseline_preds: Optional[np.ndarray] = None,
    baseline_r: Optional[float] = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
):
    residuals = z_true_c - z_pred_c
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((z_true_c - np.mean(z_true_c)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    st.markdown(f"### Statistics — {channel_name}")

    if baseline_preds is not None:
        base_residuals = z_true_c - baseline_preds
        base_mae = np.mean(np.abs(base_residuals))
        base_rmse = np.sqrt(np.mean(base_residuals**2))
        base_ss_res = np.sum(base_residuals**2)
        base_r_squared = 1 - (base_ss_res / ss_tot) if ss_tot != 0 else 0
        base_r = baseline_r if baseline_r is not None else np.nan

        st.markdown(f"**{model_name}**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Pearson r", f"{r_ch:.4f}")
        with c2:
            st.metric("R²", f"{r_squared:.4f}")
        with c3:
            st.metric("RMSE", f"{rmse:.3f}")
        with c4:
            st.metric("MAE", f"{mae:.3f}")

        st.markdown(f"**{baseline_name}**")
        c1b, c2b, c3b, c4b = st.columns(4)
        with c1b:
            st.metric("Pearson r", f"{base_r:.4f}" if not np.isnan(base_r) else "N/A")
        with c2b:
            st.metric("R²", f"{base_r_squared:.4f}")
        with c3b:
            st.metric("RMSE", f"{base_rmse:.3f}")
        with c4b:
            st.metric("MAE", f"{base_mae:.3f}")
    else:
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
    chunk_margin: float = 0.0,
    zp_1: Optional[np.ndarray] = None,
    zp_2: Optional[np.ndarray] = None,
    r_zp1: Optional[float] = None,
    r_zp2: Optional[float] = None,
    baseline_preds: Optional[np.ndarray] = None,
    baseline_name: str = "Baseline",
    baseline_r: Optional[float] = None,
):
    nz_chan = z_true.shape[1] if z_true.ndim == 2 else 1
    z_true = transpose_if_needed(z_true, len(t_abs))
    z_pred = transpose_if_needed(z_pred, len(t_abs))

    z_true_c = z_true.squeeze() if nz_chan == 1 else z_true[:, channel_idx]
    z_pred_c = z_pred.squeeze() if nz_chan == 1 else z_pred[:, channel_idx]

    z_pred_c_rescaled = rescale_to_reference(z_pred_c, z_true_c)

    onset_time = t_abs.min() if len(t_abs) > 0 else 0.0
    fig = create_base_time_series_figure(
        time_abs=t_abs,
        onset_time=onset_time,
        y_label="Value",
        title="",
    )

    fig.add_trace(
        go.Scatter(
            x=t_abs,
            y=z_true_c,
            name="Z_true",
            mode="lines",
            line=dict(color=PLOT_COLOR.stim_off, width=PLOT_STYLE.line_width_normal),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_abs,
            y=z_pred_c_rescaled,
            name="Z_pred (rescaled)",
            mode="lines",
            line=dict(
                color=PLOT_COLOR.stim_on, width=PLOT_STYLE.line_width_normal, dash="dot"
            ),
        )
    )

    if zp_1 is not None and r_zp1 is not None:
        zp_1_rescaled = rescale_to_reference(zp_1, z_true_c)
        fig.add_trace(
            go.Scatter(
                x=t_abs,
                y=zp_1_rescaled,
                name=f"Zp_1 (beh) r={r_zp1:.3f}",
                mode="lines",
                line=dict(color=PALETTE.twilight_indigo, width=PLOT_STYLE.line_width_normal, dash="dashdot"),
            )
        )

    if zp_2 is not None and r_zp2 is not None:
        zp_2_rescaled = rescale_to_reference(zp_2, z_true_c)
        fig.add_trace(
            go.Scatter(
                x=t_abs,
                y=zp_2_rescaled,
                name=f"Zp_2 (non-beh) r={r_zp2:.3f}",
                mode="lines",
                line=dict(color=PALETTE.strawberry_red, width=PLOT_STYLE.line_width_normal, dash="dot"),
            )
        )

    if baseline_preds is not None:
        baseline_rescaled = rescale_to_reference(baseline_preds, z_true_c)
        fig.add_trace(
            go.Scatter(
                x=t_abs,
                y=baseline_rescaled,
                name=f"{baseline_name} (rescaled)",
                mode="lines",
                line=dict(color=BASELINE_COLOR, width=PLOT_STYLE.line_width_normal, dash="dash"),
            )
        )

    if chunk_margin > 0:
        add_margin_visualization(fig, t_abs, chunk_margin)

    st.plotly_chart(fig, use_container_width=True)
    r_str = f"{r_ch:.3f}" if not np.isnan(r_ch) else "N/A"
    
    caption_parts = [f"Behavioral Prediction: {channel_name} (Pearson r={r_str})"]
    if baseline_preds is not None:
        baseline_r_str = f"{baseline_r:.3f}" if baseline_r is not None and not np.isnan(baseline_r) else "N/A"
        caption_parts.append(f"{baseline_name} r={baseline_r_str}")
    caption_parts.append("*Predictions rescaled to match Z_true mean/std for visualization*")
    st.caption(" | ".join(caption_parts))


def render_prediction_psd_analysis(y_true, y_pred, sampling_rate=60, channel_name="",
                                    baseline_preds=None, baseline_name="Baseline",
                                    model_name="Model"):
    if y_true is None or y_pred is None:
        return

    from dashboard.backbone import create_base_psd_line_figure

    freqs_t, psd_t = compute_power_spectrum(y_true, sampling_rate)
    freqs_p, psd_p = compute_power_spectrum(y_pred, sampling_rate)

    psd_t_val = psd_t.squeeze() if psd_t.ndim > 1 else psd_t
    psd_p_val = psd_p.squeeze() if psd_p.ndim > 1 else psd_p

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
                    color=BASELINE_COLOR, width=PLOT_STYLE.line_width_normal, dash="dash"
                ),
            )
        )

    st.plotly_chart(fig, use_container_width=True)


def render_predictions_tab(split_res: Dict[str, Any], trial_idx: int, cfg_path: Path, run_ts: str):
    Y_true = split_res["Y"]
    Yp = split_res["Yp"]
    Zp = split_res["Zp"]
    pearson_tr = split_res["pearson_per_channel"]

    y_t = np.array(Y_true[trial_idx])
    y_p = np.array(Yp[trial_idx])
    z_p = None if Zp[trial_idx] is None else np.array(Zp[trial_idx])

    offsets = split_res.get("offset", [])
    t_offset = (
        float(offsets[trial_idx])
        if offsets and len(offsets) > trial_idx and offsets[trial_idx] is not None
        else 0.0
    )
    n_samples = y_t.shape[0]
    t_abs = get_trial_time_axis(split_res, trial_idx, n_samples, t_offset)

    chunk_margin_val = split_res.get("chunk_margin", 0.0)
    if isinstance(chunk_margin_val, list):
        chunk_margin = (
            float(chunk_margin_val[trial_idx])
            if trial_idx < len(chunk_margin_val)
            else 0.0
        )
    else:
        chunk_margin = float(chunk_margin_val) if chunk_margin_val is not None else 0.0
    pearson_tr_trial = pearson_tr[trial_idx] if pearson_tr else []
    r_list = pearson_tr_trial

    if r_list:
        valid_r = [r for r in r_list if not (r is None or np.isnan(r))]
        r_mean = np.mean(valid_r) if len(valid_r) > 0 else np.nan
    else:
        r_mean = np.nan

    chan_names = split_res.get("input_channels", [])
    behavioral_input_names = split_res.get("behavioral_input_channels", []) or []
    if behavioral_input_names:
        chan_names = list(chan_names) + list(behavioral_input_names)
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

    from dashboard.subtabs.helpers import (
        list_variants, load_precomputed_results, list_run_timestamps,
        variant_short_name, find_baseline_variants, get_project_root,
    )

    project_root = get_project_root(cfg_path)
    results_root = project_root / "results"
    all_variants = list_variants(results_root)
    
    current_variant = cfg_path.stem

    baseline_variants = find_baseline_variants(current_variant, all_variants)

    baseline_yp_c = None
    baseline_r = None
    baseline_info = None
    selected_baseline_name = "Baseline"
    model_label = variant_short_name(current_variant) if cfg_path else "Model"

    if baseline_variants:
        st.markdown("#### Baseline Comparison")
        
        default_idx = 0
        
        selected_baseline = st.selectbox(
            "Select Baseline Model",
            options=["None"] + baseline_variants,
            index=default_idx + 1 if baseline_variants else 0,
            key="pred_baseline_select",
        )

        if selected_baseline != "None":
            baseline_dir = results_root / selected_baseline
            baseline_timestamps = list_run_timestamps(baseline_dir)

            if baseline_timestamps:
                baseline_ts = st.selectbox(
                    "Baseline run timestamp",
                    options=baseline_timestamps,
                    index=len(baseline_timestamps) - 1,
                    key="pred_baseline_ts_select",
                )

                # Load baseline results for same split
                split_name = st.session_state.get("pred_split", "val")
                baseline_res = load_precomputed_results(
                    baseline_dir, baseline_ts, split_name
                )

                if baseline_res and trial_idx < len(baseline_res.get("Yp", [])):
                    baseline_yp = baseline_res["Yp"][trial_idx]
                    if baseline_yp is not None:
                        baseline_yp = np.array(baseline_yp)
                        baseline_yp = transpose_if_needed(baseline_yp, len(t_abs))
                        n_baseline_chan = (
                            baseline_yp.shape[1] if baseline_yp.ndim == 2 else 1
                        )
                        if c < n_baseline_chan:
                            baseline_yp_c = (
                                baseline_yp.squeeze()
                                if n_baseline_chan == 1
                                else baseline_yp[:, c]
                            )
                            # Compute baseline correlation
                            try:
                                baseline_r = np.corrcoef(
                                    y_true_c.flatten(), baseline_yp_c.flatten()
                                )[0, 1]
                            except:
                                baseline_r = np.nan
                                
                    selected_baseline_name = variant_short_name(selected_baseline)
                    st.session_state["baseline_res_cache"] = baseline_res

                # Load baseline metadata for info display
                metadata_path = baseline_dir / f"model_{baseline_ts}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        baseline_info = json.load(f)

    st.markdown("#### Time Series: Y_true vs Y_pred")

    model_variant_name = model_label

    onset_time = t_abs.min() if len(t_abs) > 0 else 0.0
    fig_ts = create_base_time_series_figure(
        time_abs=t_abs,
        onset_time=onset_time,
        y_label="Amplitude (µV)",
        title="",
    )
    fig_ts.add_trace(
        go.Scatter(
            x=t_abs,
            y=y_true_c,
            name="True",
            mode="lines",
            line=dict(color=PLOT_COLOR.stim_off, width=PLOT_STYLE.line_width_normal),
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=t_abs,
            y=y_pred_c,
            name=model_variant_name,
            mode="lines",
            line=dict(
                color=PLOT_COLOR.stim_on, width=PLOT_STYLE.line_width_normal, dash="dot"
            ),
        )
    )

    if baseline_yp_c is not None:
        fig_ts.add_trace(
            go.Scatter(
                x=t_abs,
                y=baseline_yp_c,
                name=selected_baseline_name,
                mode="lines",
                line=dict(
                    color=BASELINE_COLOR,
                    width=PLOT_STYLE.line_width_normal,
                    dash="dash",
                ),
            )
        )

    r_str = f"{r_ch:.3f}" if not np.isnan(r_ch) else "N/A"
    baseline_r_str = (
        f"{baseline_r:.3f}"
        if baseline_r is not None and not np.isnan(baseline_r)
        else "N/A"
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    caption = f"Neural Signal Prediction: {selected_name} ({model_label} r={r_str}"
    if baseline_yp_c is not None:
        caption += f", {selected_baseline_name} r={baseline_r_str})"
    else:
        caption += ")"
    st.caption(caption)

    from utils.config import get_config

    cfg = get_config(str(cfg_path))
    fs = getattr(cfg.data, "sampling_frequency", 60)

    st.markdown("#### PSD Analysis: True vs Predicted")
    render_prediction_psd_analysis(
        y_true_c, y_pred_c, sampling_rate=fs, channel_name=selected_name,
        baseline_preds=baseline_yp_c, baseline_name=selected_baseline_name,
        model_name=model_label,
    )

    st.markdown("#### Scatter Plot: True vs Predicted")
    render_y_scatter_plot(
        y_true_c, y_pred_c, selected_name, r_ch,
        baseline_preds=baseline_yp_c,
        baseline_name=selected_baseline_name,
        model_name=model_label,
    )

    st.markdown("#### Residual Plot: Prediction Errors Over Time")
    render_y_residual_plot(
        y_true_c, 
        y_pred_c, 
        t_abs, 
        selected_name, 
        chunk_margin=chunk_margin,
        baseline_preds=baseline_yp_c,
        baseline_name=selected_baseline_name
    )

    render_statistics_table(y_true_c, y_pred_c, r_ch, selected_name)

    st.markdown("---")
    with st.expander("Residual Diagnostics & Normality Tests", expanded=False):
        render_residual_diagnostics(
            y_true_c, y_pred_c, selected_name,
            baseline_preds=baseline_yp_c,
            baseline_name=selected_baseline_name,
            model_name=model_label,
        )

    st.markdown("---")
    with st.expander("Learned Noise Covariance Matrices", expanded=False):
        if "config_path" in st.session_state and "run_timestamp" in st.session_state:
            render_learned_noise_diagnostics(
                str(st.session_state["config_path"]), st.session_state["run_timestamp"]
            )
        else:
            st.info("Model configuration not available in session state")

    Z_true = split_res.get("Z")
    
    tester = Tester.from_config_file(str(cfg_path), run_timestamp=run_ts)
    tester._load_model_for_run()
    idSys = getattr(tester.framework.model, "idSys", None)
    B_z = getattr(idSys, "B_z", None) if idSys else None
    d_z = getattr(idSys, "d_z", None) if idSys else None
    
    cfg = get_config(str(cfg_path))
    n1 = getattr(cfg.model, "n1", 0)

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

            baseline_zp_c = None
            baseline_r_z = None
            baseline_name_z = selected_baseline_name
            baseline_res_cache = st.session_state.get("baseline_res_cache")

            if baseline_res_cache and trial_idx < len(baseline_res_cache.get("Zp", [])):
                baseline_zp = baseline_res_cache["Zp"][trial_idx]
                if baseline_zp is not None:
                    baseline_zp = np.array(baseline_zp)
                    baseline_zp = transpose_if_needed(baseline_zp, len(t_abs))
                    n_baseline_chan_z = baseline_zp.shape[1] if baseline_zp.ndim == 2 else 1
                    if z_c < n_baseline_chan_z:
                        baseline_zp_c = baseline_zp.squeeze() if n_baseline_chan_z == 1 else baseline_zp[:, z_c]
                        try:
                            baseline_r_z = np.corrcoef(
                                z_true_c.flatten(), baseline_zp_c.flatten()
                            )[0, 1]
                        except:
                            baseline_r_z = np.nan

            Xp_trial = np.array(split_res.get("Xp", [])[trial_idx]) if split_res.get("Xp") and len(split_res["Xp"]) > trial_idx else None
            zp_1, zp_2, r_zp1, r_zp2 = None, None, None, None
            if Xp_trial is not None and B_z is not None and n1 > 0:
                Xp_transposed = transpose_if_needed(Xp_trial, len(t_abs))
                zp_1, zp_2, r_zp1, r_zp2 = _compute_zp_components(z_true_c, Xp_transposed, B_z, d_z, n1, z_c)

            st.markdown("#### Time Series: Z_true vs Z_pred")
            render_z_prediction_plot(
                z_t, z_p, t_abs, z_c, selected_z_name, r_z_ch,
                zp_1=zp_1, zp_2=zp_2, r_zp1=r_zp1, r_zp2=r_zp2,
                baseline_preds=baseline_zp_c, baseline_name=baseline_name_z,
                baseline_r=baseline_r_z,
            )

            st.markdown("#### PSD Analysis: Z_true vs Z_pred")
            render_prediction_psd_analysis(
                z_true_c, z_pred_c, sampling_rate=fs, channel_name=selected_z_name,
                baseline_preds=baseline_zp_c, baseline_name=baseline_name_z,
                model_name=model_label,
            )

            st.markdown("#### Scatter Plot: True vs Predicted")
            render_z_scatter_plot(
                z_true_c, z_pred_c, selected_z_name, r_z_ch,
                baseline_preds=baseline_zp_c,
                baseline_name=baseline_name_z,
                model_name=model_label,
            )

            st.markdown("#### Residual Plot: Prediction Errors Over Time")
            render_z_residual_plot(
                z_true_c, 
                z_pred_c, 
                t_abs, 
                selected_z_name, 
                baseline_preds=baseline_zp_c,
                baseline_name=baseline_name_z
            )

            render_z_statistics_table(
                z_true_c, z_pred_c, r_z_ch, selected_z_name,
                baseline_preds=baseline_zp_c,
                baseline_r=baseline_r_z,
                baseline_name=baseline_name_z,
                model_name=model_label,
            )

            st.markdown("---")
            with st.expander("Residual Diagnostics & Normality Tests", expanded=False):
                st.markdown(
                    "Comprehensive diagnostics to verify that residuals follow a Gaussian distribution."
                )
                render_residual_diagnostics(
                    z_true_c, z_pred_c, selected_z_name,
                    baseline_preds=baseline_zp_c,
                    baseline_name=baseline_name_z,
                    model_name=model_label,
                )

            st.markdown("---")
            with st.expander("Detrended Z_pred Diagnostics", expanded=False):
                z_pred_detrended = z_pred_c - np.mean(z_pred_c)
                z_true_zero = np.zeros_like(z_pred_detrended)
                base_detrended = None
                if baseline_zp_c is not None:
                    base_detrended = baseline_zp_c - np.mean(baseline_zp_c)
                render_residual_diagnostics(
                    z_true_zero, z_pred_detrended, f"{selected_z_name} (detrended)",
                    baseline_preds=base_detrended,
                    baseline_name=baseline_name_z,
                    model_name=model_label,
                )
