import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any
from scipy.ndimage import gaussian_filter

from dashboard.subtabs.helpers import get_trial_time_axis
from dashboard.backbone import (
    create_base_time_series_figure,
    PLOT_STYLE,
    PALETTE,
    PLOT_COLOR,
    add_margin_visualization,
)
from utils.config import get_config
from scipy.signal import stft
from utils.stats import (
    compute_power_spectrum,
    find_dominant_frequencies,
    spectral_correlation,
)

SAMPLING_FREQ = 60


def render_latent_states_plot(
    x_p: np.ndarray,
    t_abs: np.ndarray,
    t_offset: float,
    chunk_margin: Optional[float],
    duration: Optional[float],
    trial_idx: int,
):
    t_x = (
        np.linspace(t_abs[0], t_abs[-1], x_p.shape[0])
        if len(t_abs) != x_p.shape[0]
        else t_abs
    )
    nx = x_p.shape[1] if x_p.ndim == 2 else 1

    onset_time = t_x.min() if len(t_x) > 0 else 0.0

    fig = create_base_time_series_figure(
        time_abs=t_x,
        onset_time=onset_time,
        y_label="Raw value",
        title="",
    )

    for d in range(nx):
        series = x_p[:, d] if nx > 1 else x_p.squeeze()
        fig.add_trace(
            go.Scatter(
                x=t_x,
                y=series,
                name=f"X_p[{d}]",
                mode="lines",
                line=dict(width=PLOT_STYLE.line_width_normal),
            )
        )

    if duration is not None:
        event_start = onset_time
        event_end = (
            t_offset
            + float(duration)
            - (float(chunk_margin) if chunk_margin is not None else 0.0)
        )
        fig.add_vline(
            x=event_start,
            line_dash="dash",
            line_color=PALETTE.twilight_indigo,
            annotation_text="Event Start",
            annotation_font=dict(size=10, color=PALETTE.twilight_indigo),
        )
        fig.add_vline(
            x=event_end,
            line_dash="dash",
            line_color=PALETTE.vintage_grape,
            annotation_text="Event End",
            annotation_font=dict(size=10, color=PALETTE.vintage_grape),
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        margin=dict(l=60, r=80, t=20, b=60),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"latent_ts_{trial_idx}")
    st.caption(f"Latent States (X_p) — Trial {trial_idx} — {nx} dimensions")


def render_auxiliary_predictions_plot(
    z_p: np.ndarray,
    t_abs: np.ndarray,
    t_offset: float,
    chunk_margin: Optional[float],
    duration: Optional[float],
    trial_idx: int,
):
    t_z = (
        np.linspace(t_abs[0], t_abs[-1], z_p.shape[0])
        if len(t_abs) != z_p.shape[0]
        else t_abs
    )
    nz = z_p.shape[1] if z_p.ndim == 2 else 1

    onset_time = t_z.min() if len(t_z) > 0 else 0.0

    fig = create_base_time_series_figure(
        time_abs=t_z,
        onset_time=onset_time,
        y_label="Value",
        title="",
    )

    for d in range(nz):
        series = z_p[:, d] if nz > 1 else z_p.squeeze()
        fig.add_trace(
            go.Scatter(
                x=t_z,
                y=series,
                name=f"Z_p[{d}]",
                mode="lines",
                line=dict(width=PLOT_STYLE.line_width_normal),
            )
        )

    if duration is not None:
        event_start = onset_time
        event_end = (
            t_offset
            + float(duration)
            - (float(chunk_margin) if chunk_margin is not None else 0.0)
        )
        fig.add_vline(
            x=event_start,
            line_dash="dash",
            line_color=PALETTE.twilight_indigo,
            annotation_text="Event Start",
            annotation_font=dict(size=10, color=PALETTE.twilight_indigo),
        )
        fig.add_vline(
            x=event_end,
            line_dash="dash",
            line_color=PALETTE.vintage_grape,
            annotation_text="Event End",
            annotation_font=dict(size=10, color=PALETTE.vintage_grape),
        )

    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        margin=dict(l=60, r=80, t=20, b=60),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"aux_z_ts_{trial_idx}")
    st.caption(
        f"Auxiliary Variable Predictions (Z_p) — Trial {trial_idx} — {nz} dimensions"
    )


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
    st.markdown("#### Trajectory")
    render_trajectory(x_p, dim_x, dim_y, split_res, trial_idx, z_p=z_p)

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
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        xaxis_title=f"Latent Dimension {dim_x+1}",
        yaxis_title=f"Latent Dimension {dim_y+1}",
        showlegend=False,
        hovermode="closest",
        height=600,
        margin=dict(l=60, r=20, t=20, b=60),
    )

    st.plotly_chart(
        fig, use_container_width=True, key=f"phase_heatmap_{trial_idx}_{dim_x}_{dim_y}"
    )
    st.caption(
        f"Phase Space Heatmap: Dimension {dim_x+1} vs Dimension {dim_y+1} — Trial {trial_idx}"
    )


def _render_dbs_comparison(
    split_res: Dict[str, Any], dim_x: int, dim_y: int, trial_idx: int
):

    Xp_list = split_res.get("Xp", [])
    stim_list = split_res.get("stim", [])

    if not Xp_list or not stim_list:
        st.warning("DBS stimulation data not available")
        return

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
            template="plotly_white",
            font=dict(family=PLOT_STYLE.font_family),
            xaxis_title=f"Latent Dimension {dim_x+1}",
            yaxis_title=f"Latent Dimension {dim_y+1}",
            height=600,
            margin=dict(l=60, r=20, t=20, b=60),
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
            template="plotly_white",
            font=dict(family=PLOT_STYLE.font_family),
            xaxis_title=f"Latent Dimension {dim_x+1}",
            yaxis_title=f"Latent Dimension {dim_y+1}",
            height=600,
            margin=dict(l=60, r=20, t=20, b=60),
        )

    st.plotly_chart(fig, use_container_width=True, key=f"dbs_compare_{trial_idx}")
    st.caption(
        f"DBS Phase Space Comparison: Dim {dim_x+1} vs Dim {dim_y+1} — Trial {trial_idx}"
    )

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
    sampling_freq: float = SAMPLING_FREQ,
):
    st.markdown("### Spectral Analysis")
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
    x_signal, y_signal = x_signal[:min_len], y_signal[:min_len]

    freqs_x, psd_x = compute_power_spectrum(x_signal, sampling_freq)
    freqs_y, psd_y = compute_power_spectrum(y_signal, sampling_freq)

    psd_x_db = 10 * np.log10(psd_x.flatten() + 1e-20) + 120
    psd_y_db = 10 * np.log10(psd_y.flatten() + 1e-20) + 120

    nperseg = min(len(x_signal), 128)
    if nperseg < 8:
        nperseg = len(x_signal)

    fx, tx, Zxx_x = stft(x_signal, fs=sampling_freq, nperseg=nperseg)
    fy, ty, Zxx_y = stft(y_signal, fs=sampling_freq, nperseg=nperseg)

    spec_x = 10 * np.log10(np.abs(Zxx_x) + 1e-20) + 120
    spec_y = 10 * np.log10(np.abs(Zxx_y) + 1e-20) + 120

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "PSD Overlay",
            f"Latent Dim {latent_dim+1}",
            f"Neural Chan {neural_chan+1}",
        ],
        column_widths=[0.4, 0.3, 0.3],
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=freqs_x,
            y=psd_x_db,
            name="Latent",
            line=dict(color=PALETTE.twilight_indigo, width=1.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=freqs_y,
            y=psd_y_db,
            name="Neural",
            line=dict(color=PALETTE.strawberry_red, width=1.2, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=spec_x, x=tx, y=fx, colorscale="Viridis", showscale=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(z=spec_y, x=ty, y=fy, colorscale="Viridis", showscale=False),
        row=1,
        col=3,
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        height=400,
        margin=dict(l=60, r=20, t=40, b=60),
    )
    fig.update_xaxes(title_text="Freq (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Power (dB)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Freq (Hz)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=1, col=3)

    st.plotly_chart(fig, use_container_width=True, key=f"latent_spectral_{trial_idx}")
    st.caption(
        f"Spectral analysis: Latent Dim {latent_dim+1} vs Neural Chan {neural_chan+1} (Dashed)"
    )


def render_trajectory(
    x_p: np.ndarray,
    dim_x: int,
    dim_y: int,
    split_res: Dict[str, Any],
    trial_idx: int,
    z_p: Optional[np.ndarray] = None,
    step: int = 10,
):
    x, y = x_p[:, dim_x], x_p[:, dim_y]
    dx, dy = np.diff(x), np.diff(y)

    t_abs = (
        split_res.get("time_abs", [None])[trial_idx]
        if trial_idx < len(split_res.get("time_abs", []))
        else None
    )

    if z_p is not None and z_p.ndim == 2 and z_p.shape[1] > 1:
        z_channel_options = split_res.get(
            "output_channels", [f"z_ch{i}" for i in range(z_p.shape[1])]
        )
        selected_z_ch = st.selectbox(
            "Color trajectory by:",
            options=range(z_p.shape[1]),
            format_func=lambda i: (
                z_channel_options[i] if i < len(z_channel_options) else f"Z[{i}]"
            ),
            key=f"traj_color_z_{trial_idx}",
        )
        color_values = z_p[:, selected_z_ch]
        color_label = z_channel_options[selected_z_ch]
    else:
        if z_p is not None and z_p.size > 0:
            output_chan_names = split_res.get("output_channels", [])
            color_values = z_p[:, 0] if z_p.ndim == 2 else z_p
            color_label = output_chan_names[0] if output_chan_names else "Z_p"
        elif t_abs is not None:
            color_values = t_abs
            color_label = "Time (s)"
        else:
            color_values = np.arange(len(x))
            color_label = "Sample Index"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="rgba(56, 64, 95, 0.3)", width=0.8),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Colored markers based on selection
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=3.5,
                color=color_values,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title=color_label,
                    thickness=15,
                    len=0.7,
                    title_font=dict(size=10),
                    tickfont=dict(size=9),
                ),
            ),
            showlegend=False,
            hovertemplate=f"<b>Time</b>: %{{text}}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>{color_label}: %{{marker.color:.3f}}<extra></extra>",
            text=(
                [f"{t:.2f}s" for t in t_abs]
                if t_abs is not None
                else [f"idx {i}" for i in range(len(x))]
            ),
        )
    )

    arrow_step = max(3, len(x) // 35)
    arrow_indices = np.arange(0, len(dx), arrow_step)

    for i in arrow_indices:
        if i + 1 < len(x):
            fig.add_annotation(
                x=x[i + 1],
                y=y[i + 1],
                ax=x[i],
                ay=y[i],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=1,
                arrowsize=0.8,
                arrowwidth=0.8,
                arrowcolor=PALETTE.twilight_indigo,
                opacity=0.6,
            )
    fig.add_trace(
        go.Scatter(
            x=[x[0]],
            y=[y[0]],
            mode="markers",
            marker=dict(
                size=10,
                color="white",
                symbol="circle",
                line=dict(width=1.5, color=PALETTE.cool_steel),
            ),
            name="Start",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[x[-1]],
            y=[y[-1]],
            mode="markers",
            marker=dict(
                size=10,
                color=PALETTE.strawberry_red,
                symbol="circle",
                line=dict(width=1.5, color="white"),
            ),
            name="End",
        )
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        xaxis_title=f"Latent Dimension {dim_x+1}",
        yaxis_title=f"Latent Dimension {dim_y+1}",
        height=600,
        margin=dict(l=60, r=20, t=20, b=60),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"trajectory_{trial_idx}")
    st.caption(f"Trajectory: Dim {dim_x+1} vs Dim {dim_y+1} — Trial {trial_idx}")


def render_latent_states_tab(split_res: Dict[str, Any], trial_idx: int):
    Xp = split_res.get("Xp", [])
    Zp = split_res.get("Zp", [])

    xp_trial = Xp[trial_idx] if Xp and len(Xp) > trial_idx else None
    x_p = np.array(xp_trial) if xp_trial is not None else None

    zp_trial = Zp[trial_idx] if Zp and len(Zp) > trial_idx else None
    z_p = np.array(zp_trial) if zp_trial is not None else None

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

    else:
        st.info("No latent states (Xp) available for this trial. ")
