import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from dashboard.backbone import PALETTE, PLOT_STYLE
from utils.stats import (
    compute_session_cross_correlation,
    compute_average_cross_correlation,
)


def render_lag_consistency_heatmap(
    session_result: Dict[str, Any],
    title: str = "Lag Consistency Heatmap",
):
    lags_ms = session_result["lags_ms"]
    correlations = session_result["correlations"]
    n_trials = correlations.shape[0]

    y_labels = [f"Trial {i}" for i in range(n_trials)]

    fig = go.Figure(
        data=go.Heatmap(
            z=correlations,
            x=lags_ms,
            y=y_labels,
            colorscale="Viridis",
            colorbar=dict(title="R"),
            zmin=-0.5,
            zmax=1.0,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Lag (ms)",
        yaxis_title="Trial",
        height=max(400, 250 + n_trials * 18),
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family, size=12),
    )

    return fig


def render_summary_cross_correlation_plot(
    avg_result: Dict[str, Any],
    title: str = "Summary Cross-Correlation",
):
    lags_ms = avg_result["lags_ms"]
    mean_corr = avg_result["mean_correlation"]
    median_corr = avg_result.get("median_correlation", mean_corr)
    std_corr = avg_result["std_correlation"]

    upper = mean_corr + std_corr
    lower = mean_corr - std_corr

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([lags_ms, lags_ms[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(100, 149, 237, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±1 SD (Mean)",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=lags_ms,
            y=mean_corr,
            mode="lines",
            name="Mean R",
            line=dict(color=PALETTE.vintage_grape, width=1.5, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=lags_ms,
            y=median_corr,
            mode="lines",
            name="Median R",
            line=dict(color=PALETTE.vintage_grape, width=3),
        )
    )

    peak_lag = avg_result.get("median_optimal_lag_ms", avg_result["avg_optimal_lag_ms"])
    peak_r = np.nanmax(median_corr) if not np.all(np.isnan(median_corr)) else np.nan
    lag_std = avg_result["lag_std_ms"]

    if not np.isnan(peak_lag):
        fig.add_vline(
            x=peak_lag,
            line_dash="dash",
            line_color=PALETTE.strawberry_red,
            line_width=2,
            annotation_text=f"Median Peak: {peak_lag:.0f}ms ± {lag_std:.0f}ms",
            annotation_position="top right",
            annotation_font=dict(size=11, color=PALETTE.strawberry_red),
        )

    fig.update_layout(
        title=title,
        xaxis_title="Lag (ms)",
        yaxis_title="Correlation (R)",
        height=400,
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def render_cross_correlation_analysis_tab(
    split_res: Dict[str, Any],
    sampling_freq: float = 60.0,
):
    st.subheader("Cross-Correlation Lag Analysis")
    st.markdown(
        "Analyzes temporal alignment between predictions/forecasts and true signals."
    )

    col1, col2 = st.columns(2)
    with col1:
        max_lag_ms = st.slider("Max Lag (ms)", 100, 1000, 500, 50, key="xcorr_max_lag")
    with col2:
        min_lag_ms = st.slider("Min Lag (ms)", 0, 200, 0, 10, key="xcorr_min_lag")

    Y_true = split_res.get("Y", [])
    Yp = split_res.get("Yp", [])
    Y_future_true = split_res.get("Y_future_true", [])
    Y_future_pred = split_res.get("Y_future_pred", [])

    Z_true = split_res.get("Z", [])
    Zp = split_res.get("Zp", [])
    Z_future_true = split_res.get("Z_future_true", [])
    Z_future_pred = split_res.get("Z_future_pred", [])

    has_y_pred = Y_true and Yp and len(Y_true) > 0 and len(Yp) > 0
    has_y_fore = Y_future_true and Y_future_pred and len(Y_future_true) > 0
    has_z_pred = Z_true and Zp and len(Z_true) > 0 and len(Zp) > 0
    has_z_fore = Z_future_true and Z_future_pred and len(Z_future_true) > 0

    if not any([has_y_pred, has_y_fore, has_z_pred, has_z_fore]):
        st.info(
            "No prediction or forecast data available for cross-correlation analysis."
        )
        return

    tabs = []
    tab_names = []

    if has_y_pred:
        tab_names.append("Neural Prediction")
    if has_y_fore:
        tab_names.append("Neural Forecast")
    if has_z_pred:
        tab_names.append("Behavioral Prediction")
    if has_z_fore:
        tab_names.append("Behavioral Forecast")

    if len(tab_names) == 0:
        return

    tabs = st.tabs(tab_names)
    tab_idx = 0

    if has_y_pred:
        with tabs[tab_idx]:
            _render_xcorr_section(
                Y_true,
                Yp,
                sampling_freq,
                max_lag_ms,
                min_lag_ms,
                "Neural Prediction",
                "neural_pred",
            )
        tab_idx += 1

    if has_y_fore:
        with tabs[tab_idx]:
            valid_fore_true = [arr for arr in Y_future_true if arr is not None]
            valid_fore_pred = [arr for arr in Y_future_pred if arr is not None]
            if valid_fore_true and valid_fore_pred:
                _render_xcorr_section(
                    valid_fore_true,
                    valid_fore_pred,
                    sampling_freq,
                    max_lag_ms,
                    min_lag_ms,
                    "Neural Forecast",
                    "neural_fore",
                )
            else:
                st.info("No valid forecast data available.")
        tab_idx += 1

    if has_z_pred:
        with tabs[tab_idx]:
            valid_z = [arr for arr in Z_true if arr is not None]
            valid_zp = [arr for arr in Zp if arr is not None]
            if valid_z and valid_zp:
                _render_xcorr_section(
                    valid_z,
                    valid_zp,
                    sampling_freq,
                    max_lag_ms,
                    min_lag_ms,
                    "Behavioral Prediction",
                    "behav_pred",
                )
            else:
                st.info("No valid behavioral prediction data.")
        tab_idx += 1

    if has_z_fore:
        with tabs[tab_idx]:
            valid_zf_true = [arr for arr in Z_future_true if arr is not None]
            valid_zf_pred = [arr for arr in Z_future_pred if arr is not None]
            if valid_zf_true and valid_zf_pred:
                _render_xcorr_section(
                    valid_zf_true,
                    valid_zf_pred,
                    sampling_freq,
                    max_lag_ms,
                    min_lag_ms,
                    "Behavioral Forecast",
                    "behav_fore",
                )
            else:
                st.info("No valid behavioral forecast data.")


def _render_xcorr_section(
    true_list,
    pred_list,
    sampling_freq,
    max_lag_ms,
    min_lag_ms,
    title_prefix,
    key_prefix,
):
    session_result = compute_session_cross_correlation(
        true_list, pred_list, sampling_freq, max_lag_ms, min_lag_ms
    )
    avg_result = compute_average_cross_correlation(session_result)

    st.markdown(f"### {title_prefix}: Lag Consistency Heatmap")
    fig_heatmap = render_lag_consistency_heatmap(
        session_result, title=f"{title_prefix} - Correlation by Trial and Lag"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True, key=f"{key_prefix}_heatmap")

    st.markdown(f"### {title_prefix}: Summary Cross-Correlation")
    fig_summary = render_summary_cross_correlation_plot(
        avg_result, title=f"{title_prefix} - Mean & Median Cross-Correlation Curves"
    )
    st.plotly_chart(fig_summary, use_container_width=True, key=f"{key_prefix}_summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Median Optimal Lag", f"{avg_result.get('median_optimal_lag_ms', 0):.1f} ms"
        )
    with col2:
        st.metric("Lag Variability (SD)", f"±{avg_result['lag_std_ms']:.1f} ms")
    with col3:
        st.metric("Mean Peak Correlation", f"{avg_result['avg_peak_correlation']:.3f}")
