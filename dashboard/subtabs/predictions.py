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
from dashboard.subtabs.helpers import (
    get_trial_time_axis,
    transpose_if_needed,
    rescale_to_reference,
)

BASELINE_COLOR = "#00E5FF"
from utils.stats import (
    compute_residual_statistics,
    qq_plot_data,
    normality_tests,
    probability_plot_data,
    whiteness_test,
    compute_power_spectrum,
    find_dominant_frequencies,
    spectral_correlation,
)
from dashboard.backbone import (
    create_base_psd_line_figure,
    render_styled_table,
)
from dashboard.subtabs.helpers import (
    list_variants,
    load_precomputed_results,
    list_run_timestamps,
    variant_short_name,
    find_baseline_variants,
    get_project_root,
    render_prediction_psd_analysis,
    render_residual_plot,
    render_statistics_table,
    render_residual_diagnostics,
    select_baseline,
    get_channel,
    get_baseline_channel,
    render_analysis,
)
from utils.config import get_config


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
            df_brain_styled = df_brain.copy()
            for col in df_brain_styled.columns:
                df_brain_styled[col] = df_brain_styled[col].apply(lambda x: f"{x:.4f}")
            render_styled_table(df_brain_styled, key="tbl_ev_brain")
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
            df_noise_styled = df_noise.copy()
            for col in df_noise_styled.columns:
                df_noise_styled[col] = df_noise_styled[col].apply(lambda x: f"{x:.4f}")
            render_styled_table(df_noise_styled, key="tbl_ev_noise")
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
                        yaxis=dict(autorange="reversed"),
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
                        yaxis=dict(autorange="reversed"),
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
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
                    st.caption(f"Shape: {S_np.shape}")
        else:
            st.info("No noise covariance matrices found in model")
    else:
        st.info("This model type does not expose noise covariance matrices")


def _compute_zp_components(
    z_true_c: np.ndarray,
    xp: np.ndarray,
    B_z: np.ndarray,
    d_z: np.ndarray,
    n1: int,
    channel_idx: int,
) -> tuple:
    if xp is None or B_z is None or n1 <= 0 or xp.shape[1] < n1:
        return None, None, np.nan, np.nan

    xp_1 = xp[:, :n1]
    xp_2 = xp[:, n1:]

    B_z_1 = B_z[channel_idx, :n1]
    B_z_2 = B_z[channel_idx, n1:]

    zp_1 = xp_1 @ B_z_1
    if d_z is not None and len(d_z) > channel_idx:
        zp_1 += d_z[channel_idx]

    zp_2 = xp_2 @ B_z_2
    # Do not add d_z to zp_2 to avoid double counting the bias

    r_zp1 = (
        np.corrcoef(z_true_c.flatten(), zp_1.flatten())[0, 1]
        if len(z_true_c) > 1 and len(zp_1) > 1
        else np.nan
    )
    r_zp2 = (
        np.corrcoef(z_true_c.flatten(), zp_2.flatten())[0, 1]
        if len(z_true_c) > 1 and len(zp_2) > 1
        else np.nan
    )

    return zp_1, zp_2, r_zp1, r_zp2


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
                line=dict(
                    color=PALETTE.twilight_indigo,
                    width=PLOT_STYLE.line_width_normal,
                    dash="dashdot",
                ),
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
                line=dict(
                    color=PALETTE.strawberry_red,
                    width=PLOT_STYLE.line_width_normal,
                    dash="dot",
                ),
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
                line=dict(
                    color=BASELINE_COLOR,
                    width=PLOT_STYLE.line_width_normal,
                    dash="dash",
                ),
            )
        )

    if chunk_margin > 0:
        add_margin_visualization(fig, t_abs, chunk_margin)

    st.plotly_chart(fig, use_container_width=True)
    r_str = f"{r_ch:.3f}" if not np.isnan(r_ch) else "N/A"

    caption_parts = [f"Behavioral Prediction: {channel_name} (Pearson r={r_str})"]
    if baseline_preds is not None:
        baseline_r_str = (
            f"{baseline_r:.3f}"
            if baseline_r is not None and not np.isnan(baseline_r)
            else "N/A"
        )
        caption_parts.append(f"{baseline_name} r={baseline_r_str}")
    caption_parts.append(
        "*Predictions rescaled to match Z_true mean/std for visualization*"
    )
    st.caption(" | ".join(caption_parts))


def render_prediction_psd_analysis(
    y_true,
    y_pred,
    sampling_rate=60,
    channel_name="",
    baseline_preds=None,
    baseline_name="Baseline",
    model_name="Model",
):
    if y_true is None or y_pred is None:
        return

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
                    color=BASELINE_COLOR,
                    width=PLOT_STYLE.line_width_normal,
                    dash="dash",
                ),
            )
        )

    st.plotly_chart(fig, use_container_width=True)


def render_predictions_tab(
    split_res: Dict[str, Any], trial_idx: int, cfg_path: Path, run_ts: str
):
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

    y_true_c = get_channel(y_t, c, t_abs)
    y_pred_c = get_channel(y_p, c, t_abs)

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

    split_name = st.session_state.get("pred_split", "val")
    baseline_res, selected_baseline_name, model_label, _baseline_variant = (
        select_baseline(cfg_path, "pred", split_name, trial_idx)
    )
    main_ch_names = split_res.get("input_channels", [])
    baseline_ch_names = baseline_res.get("input_channels", []) if baseline_res else []
    baseline_yp_c, baseline_r = get_baseline_channel(
        baseline_res,
        "Yp",
        trial_idx,
        c,
        t_abs,
        y_true_c,
        main_channel_names=main_ch_names,
        baseline_channel_names=baseline_ch_names,
    )
    if baseline_res is not None:
        st.session_state["baseline_res_cache"] = baseline_res

    st.markdown("#### Time Series: Y_true vs Y_pred")

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
            name=model_label,
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

    # --- Y analysis pipeline (shared) ---
    cfg = get_config(str(cfg_path))
    fs = getattr(cfg.data, "sampling_frequency", 80)

    render_analysis(
        y_true_c,
        y_pred_c,
        t_abs,
        selected_name,
        r_ch,
        sampling_rate=fs,
        unit="µV",
        chunk_margin=chunk_margin,
        baseline_pred_c=baseline_yp_c,
        baseline_r=baseline_r,
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

    # --- Z behavioral predictions ---
    Z_true = split_res.get("Z")

    tester = Tester.from_config_file(str(cfg_path), run_timestamp=run_ts)
    tester._load_model_for_run()
    idSys = getattr(tester.framework.model, "idSys", None)
    B_z = getattr(idSys, "B_z", None) if idSys else None
    d_z = getattr(idSys, "d_z", None) if idSys else None

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

            r_z_ch = r_z_list[z_c] if r_z_list and z_c < len(r_z_list) else np.nan

            z_true_c = get_channel(z_t, z_c, t_abs)
            z_pred_c = get_channel(z_p, z_c, t_abs)

            main_z_ch_names = split_res.get("output_channels", [])
            baseline_z_ch_names = (
                baseline_res.get("output_channels", []) if baseline_res else []
            )
            baseline_zp_c, baseline_r_z = get_baseline_channel(
                baseline_res,
                "Zp",
                trial_idx,
                z_c,
                t_abs,
                z_true_c,
                main_channel_names=main_z_ch_names,
                baseline_channel_names=baseline_z_ch_names,
            )

            Xp_trial = (
                np.array(split_res.get("Xp", [])[trial_idx])
                if split_res.get("Xp") and len(split_res["Xp"]) > trial_idx
                else None
            )
            zp_1, zp_2, r_zp1, r_zp2 = None, None, None, None
            if Xp_trial is not None and B_z is not None and n1 > 0:
                Xp_transposed = transpose_if_needed(Xp_trial, len(t_abs))
                zp_1, zp_2, r_zp1, r_zp2 = _compute_zp_components(
                    z_true_c, Xp_transposed, B_z, d_z, n1, z_c
                )

            st.markdown("#### Time Series: Z_true vs Z_pred")
            render_z_prediction_plot(
                z_t,
                z_p,
                t_abs,
                z_c,
                selected_z_name,
                r_z_ch,
                zp_1=zp_1,
                zp_2=zp_2,
                r_zp1=r_zp1,
                r_zp2=r_zp2,
                baseline_preds=baseline_zp_c,
                baseline_name=selected_baseline_name,
                baseline_r=baseline_r_z,
            )

            render_analysis(
                z_true_c,
                z_pred_c,
                t_abs,
                selected_z_name,
                r_z_ch,
                sampling_rate=fs,
                baseline_pred_c=baseline_zp_c,
                baseline_r=baseline_r_z,
                baseline_name=selected_baseline_name,
                model_name=model_label,
                rescale=True,
                show_psd=False,
            )

            st.markdown("---")
            with st.expander("Detrended Z_pred Diagnostics", expanded=False):
                z_pred_detrended = z_pred_c - np.mean(z_pred_c)
                z_true_zero = np.zeros_like(z_pred_detrended)
                base_detrended = None
                if baseline_zp_c is not None:
                    base_detrended = baseline_zp_c - np.mean(baseline_zp_c)
                render_residual_diagnostics(
                    z_true_zero,
                    z_pred_detrended,
                    f"{selected_z_name} (detrended)",
                    baseline_preds=base_detrended,
                    baseline_name=selected_baseline_name,
                    model_name=model_label,
                    rescale=True,
                    sampling_freq=fs,
                    is_neural=False,
                )
