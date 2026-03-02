import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from dashboard.backbone import (
    PALETTE,
    PLOT_STYLE,
    PLOT_COLOR,
    create_base_time_series_figure,
    add_caption_below,
)
from dashboard.subtabs.predictions import (
    _compute_zp_components,
)
from utils.config import get_config
from dashboard.subtabs.helpers import (
    render_prediction_psd_analysis,
    render_residual_plot,
    render_statistics_table,
    render_residual_diagnostics,
    BASELINE_COLOR,
    variant_short_name,
    find_baseline_variants,
    get_project_root,
    find_config_path,
    rescale_to_reference,
    list_variants,
    load_precomputed_results,
    list_run_timestamps,
    get_trial_time_axis,
    compute_forecast_for_trial,
    select_baseline,
    get_channel,
    get_baseline_channel,
    render_analysis,
)
from dashboard.backbone import render_styled_table
from utils.stats import (
    compute_power_spectrum,
    find_dominant_frequencies,
    spectral_correlation,
)

SAMPLING_FREQ = 60


def render_y_forecast_plot(
    y_concat: np.ndarray,
    y_future_true: np.ndarray,
    y_future_pred: np.ndarray,
    t_abs_margined: np.ndarray,
    m_samples: int,
    channel_idx: int,
    channel_name: str,
    r_fore_ch: float,
    baseline_yp_c: Optional[np.ndarray] = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
    baseline_r: Optional[float] = None,
):

    n_chan = y_concat.shape[1] if y_concat.ndim == 2 else 1
    y_concat_c = y_concat.squeeze() if n_chan == 1 else y_concat[:, channel_idx]
    y_ft_c = y_future_true.squeeze() if n_chan == 1 else y_future_true[:, channel_idx]
    y_fp_c = y_future_pred.squeeze() if n_chan == 1 else y_future_pred[:, channel_idx]

    # Rescale forecast to match true signal's mean/std for visualization
    y_fp_c_rescaled = rescale_to_reference(y_fp_c, y_ft_c)

    T = len(y_concat_c)
    Tpast = max(0, T - m_samples)
    t_past = t_abs_margined[:Tpast]
    t_future = t_abs_margined[Tpast:T]

    t_present_idx = max(0, Tpast - 1) if Tpast > 0 else 0
    t_present = (
        t_abs_margined[t_present_idx]
        if t_present_idx < len(t_abs_margined)
        else t_abs_margined[-1]
    )

    onset_time = t_abs_margined.min() if len(t_abs_margined) > 0 else 0.0
    fig = create_base_time_series_figure(
        time_abs=t_abs_margined,
        onset_time=onset_time,
        y_label="Amplitude (µV)",
        title="",
    )

    baseline_y_fp_c_rescaled = None
    if baseline_yp_c is not None:
        baseline_y_fp_c_rescaled = rescale_to_reference(baseline_yp_c, y_ft_c)

    if Tpast > 0:
        fig.add_trace(
            go.Scatter(
                x=t_past,
                y=y_concat_c[:Tpast],
                name="History",
                mode="lines",
                line=dict(color=PALETTE.cool_steel, width=PLOT_STYLE.line_width_normal),
            )
        )

        t_future_plot = t_abs_margined[Tpast - 1 : T]
        last_hist_val = y_concat_c[Tpast - 1]
        y_ft_plot = np.concatenate(([last_hist_val], y_ft_c))
        y_fp_plot = np.concatenate(([last_hist_val], y_fp_c_rescaled))
    else:
        t_future_plot = t_future
        y_ft_plot = y_ft_c
        y_fp_plot = y_fp_c_rescaled

    fig.add_trace(
        go.Scatter(
            x=t_future_plot,
            y=y_ft_plot,
            name="True Future",
            mode="lines",
            line=dict(color=PLOT_COLOR.stim_off, width=PLOT_STYLE.line_width_normal),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t_future_plot,
            y=y_fp_plot,
            name=f"{model_name} (rescaled)",
            mode="lines",
            line=dict(
                color=PLOT_COLOR.stim_on, width=PLOT_STYLE.line_width_normal, dash="dot"
            ),
        )
    )

    if baseline_y_fp_c_rescaled is not None:
        if Tpast > 0:
            baseline_fp_plot = np.concatenate(
                ([last_hist_val], baseline_y_fp_c_rescaled)
            )
        else:
            baseline_fp_plot = baseline_y_fp_c_rescaled
        fig.add_trace(
            go.Scatter(
                x=t_future_plot,
                y=baseline_fp_plot,
                name=f"{baseline_name} (rescaled)",
                mode="lines",
                line=dict(
                    color=BASELINE_COLOR,
                    width=PLOT_STYLE.line_width_normal,
                    dash="dash",
                ),
            )
        )

    fig.add_vline(
        x=t_present,
        line_dash="dash",
        line_color=PALETTE.vintage_grape,
        line_width=1.2,
        annotation_text="Forecast Start",
        annotation_position="top right",
        annotation_font=dict(size=10, color=PALETTE.vintage_grape),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"y_forecast_{channel_name}")
    r_str = f"{r_fore_ch:.3f}" if not np.isnan(r_fore_ch) else "N/A"
    caption_parts = [f"Neural Signal Forecast: {channel_name} ({model_name} r={r_str})"]
    if baseline_y_fp_c_rescaled is not None:
        if baseline_r is not None and not np.isnan(baseline_r):
            baseline_r_str = f"{baseline_r:.3f}"
        else:
            # Fallback: compute if not provided
            try:
                computed_r = np.corrcoef(y_ft_c.flatten(), baseline_yp_c.flatten())[
                    0, 1
                ]
                baseline_r_str = (
                    f"{computed_r:.3f}" if not np.isnan(computed_r) else "N/A"
                )
            except:
                baseline_r_str = "N/A"
        caption_parts.append(f"{baseline_name} r={baseline_r_str}")

    caption_parts.append(
        "*Forecast rescaled to match Y_true mean/std for visualization*"
    )
    st.caption(" | ".join(caption_parts))


def render_z_forecast_plot(
    z_concat: Optional[np.ndarray],
    z_future_true: np.ndarray,
    z_future_pred: np.ndarray,
    t_abs_margined: np.ndarray,
    m_samples: int,
    channel_idx: int,
    channel_name: str,
    r_fore_z_ch: float,
    zp_1: Optional[np.ndarray] = None,
    zp_2: Optional[np.ndarray] = None,
    r_zp1: Optional[float] = None,
    r_zp2: Optional[float] = None,
    baseline_zp_c: Optional[np.ndarray] = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
    baseline_r: Optional[float] = None,
):

    nz_chan = z_future_true.shape[1] if z_future_true.ndim == 2 else 1

    z_ft_c = z_future_true.squeeze() if nz_chan == 1 else z_future_true[:, channel_idx]
    z_fp_c = z_future_pred.squeeze() if nz_chan == 1 else z_future_pred[:, channel_idx]

    z_fp_c_rescaled = rescale_to_reference(z_fp_c, z_ft_c)

    T = (
        len(z_concat)
        if z_concat is not None
        else len(z_ft_c) + len(t_abs_margined) - m_samples
    )
    Tpast = max(0, len(t_abs_margined) - m_samples)
    t_past = t_abs_margined[:Tpast]
    t_future = t_abs_margined[Tpast : Tpast + len(z_ft_c)]

    t_present_idx = max(0, Tpast - 1) if Tpast > 0 else 0
    t_present = (
        t_abs_margined[t_present_idx]
        if t_present_idx < len(t_abs_margined)
        else t_abs_margined[-1]
    )

    onset_time = t_abs_margined.min() if len(t_abs_margined) > 0 else 0.0
    fig = create_base_time_series_figure(
        time_abs=t_abs_margined,
        onset_time=onset_time,
        y_label="Value",
        title="",
    )

    baseline_z_fp_c_rescaled = None
    if baseline_zp_c is not None:
        baseline_z_fp_c_rescaled = rescale_to_reference(baseline_zp_c, z_ft_c)

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
                    line=dict(
                        color=PALETTE.cool_steel, width=PLOT_STYLE.line_width_normal
                    ),
                )
            )

            t_future_plot = t_abs_margined[Tpast - 1 : Tpast + len(z_ft_c)]
            last_hist_val = z_concat_c[Tpast - 1]
            z_ft_plot = np.concatenate(([last_hist_val], z_ft_c))
            z_fp_plot = np.concatenate(([last_hist_val], z_fp_c_rescaled))
        else:
            t_future_plot = t_future
            z_ft_plot = z_ft_c
            z_fp_plot = z_fp_c_rescaled
    else:
        t_future_plot = t_future
        z_ft_plot = z_ft_c
        z_fp_plot = z_fp_c_rescaled

    fig.add_trace(
        go.Scatter(
            x=t_future_plot,
            y=z_ft_plot,
            name="True Future",
            mode="lines",
            line=dict(color=PLOT_COLOR.stim_off, width=PLOT_STYLE.line_width_normal),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_future_plot,
            y=z_fp_plot,
            name=f"{model_name} (rescaled)",
            mode="lines",
            line=dict(
                color=PLOT_COLOR.stim_on, width=PLOT_STYLE.line_width_normal, dash="dot"
            ),
        )
    )

    if baseline_z_fp_c_rescaled is not None:
        if Tpast > 0 and z_concat is not None and len(z_concat_c) >= Tpast:
            baseline_fp_plot = np.concatenate(
                ([z_concat_c[Tpast - 1]], baseline_z_fp_c_rescaled)
            )
        else:
            baseline_fp_plot = baseline_z_fp_c_rescaled
        fig.add_trace(
            go.Scatter(
                x=t_future_plot[: len(baseline_fp_plot)],
                y=baseline_fp_plot,
                name=f"{baseline_name} (rescaled)",
                mode="lines",
                line=dict(
                    color=BASELINE_COLOR,
                    width=PLOT_STYLE.line_width_normal,
                    dash="dash",
                ),
            )
        )

    if zp_1 is not None and r_zp1 is not None:
        zp_1_rescaled = rescale_to_reference(zp_1, z_ft_c)
        fig.add_trace(
            go.Scatter(
                x=(
                    t_future_plot[-len(zp_1_rescaled) :]
                    if Tpast == 0
                    else t_future_plot[1:]
                ),
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
        zp_2_rescaled = rescale_to_reference(zp_2, z_ft_c)
        fig.add_trace(
            go.Scatter(
                x=(
                    t_future_plot[-len(zp_2_rescaled) :]
                    if Tpast == 0
                    else t_future_plot[1:]
                ),
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

    fig.add_vline(
        x=t_present,
        line_dash="dash",
        line_color=PALETTE.vintage_grape,
        line_width=1.2,
        annotation_text="Forecast Start",
        annotation_position="top right",
        annotation_font=dict(size=10, color=PALETTE.vintage_grape),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"z_forecast_{channel_name}")
    r_str = f"{r_fore_z_ch:.3f}" if not np.isnan(r_fore_z_ch) else "N/A"
    caption_parts = [f"Behavioral Forecast: {channel_name} (Pearson r={r_str})"]
    if baseline_z_fp_c_rescaled is not None:
        baseline_r_str = (
            f"{baseline_r:.3f}"
            if baseline_r is not None and not np.isnan(baseline_r)
            else "N/A"
        )
        caption_parts.append(f"{baseline_name} r={baseline_r_str}")
    caption_parts.append(
        "*Forecast rescaled to match Z_true mean/std for visualization*"
    )
    st.caption(" | ".join(caption_parts))


def render_x_forecast_plot(
    x_history: np.ndarray,
    x_future_pred: np.ndarray,
    t_abs_margined: np.ndarray,
    m_samples: int,
    n1: int,
):
    nx = x_history.shape[1] if x_history.ndim == 2 else 1
    Tpast = len(x_history)
    Tfuture = len(x_future_pred)

    t_past = t_abs_margined[:Tpast]
    t_future = t_abs_margined[Tpast : Tpast + Tfuture]

    onset_time = t_abs_margined.min() if len(t_abs_margined) > 0 else 0.0
    fig = create_base_time_series_figure(
        time_abs=t_abs_margined[: Tpast + Tfuture],
        onset_time=onset_time,
        y_label="Latent Value",
        title="Latent State Forecast",
    )

    t_present = t_abs_margined[Tpast - 1] if Tpast > 0 else t_abs_margined[0]

    for d in range(nx):
        is_behavioral = d < n1
        # Use Twilight Indigo for behavioral, Strawberry Red for non-behavioral
        color = PALETTE.twilight_indigo if is_behavioral else PALETTE.strawberry_red
        name_prefix = f"X[{d}]" + (" (beh)" if is_behavioral else " (non-beh)")
        # History
        fig.add_trace(
            go.Scatter(
                x=t_past,
                y=x_history[:, d],
                name=f"{name_prefix} Hist",
                mode="lines",
                line=dict(color=color, width=PLOT_STYLE.line_width_normal),
                opacity=0.4,
                showlegend=True if d < 2 or d == n1 else False,
            )
        )

        # Forecast
        # Connect last history point to first forecast point
        t_forecast_plot = (
            np.concatenate(([t_past[-1]], t_future)) if Tpast > 0 else t_future
        )
        y_forecast_plot = (
            np.concatenate(([x_history[-1, d]], x_future_pred[:, d]))
            if Tpast > 0
            else x_future_pred[:, d]
        )

        fig.add_trace(
            go.Scatter(
                x=t_forecast_plot,
                y=y_forecast_plot,
                name=f"{name_prefix} Fore",
                mode="lines",
                line=dict(color=color, width=PLOT_STYLE.line_width_normal, dash="dash"),
                showlegend=True if d < 2 or d == n1 else False,
            )
        )

    fig.add_vline(
        x=t_present,
        line_dash="dash",
        line_color=PALETTE.vintage_grape,
        line_width=1.2,
        annotation_text="Forecast Start",
        annotation_position="top right",
    )

    st.plotly_chart(fig, use_container_width=True, key="x_forecast")
    st.caption(
        f"Latent State Forecast: {n1} behavioral (Indigo) and {nx-n1} non-behavioral (Red) dimensions."
    )


def render_forecasting_tab(
    split_res: Dict[str, Any],
    trial_idx: int,
    cfg_path: Path,
    run_ts: str,
    Y_true: List,
    Yp: List,
):

    # --- Baseline selection ---
    split_name = st.session_state.get("pred_split", "val")
    baseline_res, selected_baseline_name, model_label, baseline_variant = (
        select_baseline(cfg_path, "fore", split_name, trial_idx)
    )

    # Compute baseline forecast if needed (forecast-specific)
    baseline_forecast_res = None
    if baseline_res is not None:
        # Check if forecast results are already in baseline_res (same as main model logic)
        if (
            "Y_future_pred" in baseline_res
            and baseline_res["Y_future_pred"] is not None
        ):
            if len(baseline_res["Y_future_pred"]) > trial_idx:
                baseline_forecast_res = {}
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
                    if k in baseline_res:
                        val = baseline_res[k][trial_idx]
                        if val is not None:
                            baseline_forecast_res[k] = val

                if "metric_m" in baseline_res:
                    m_val = baseline_res["metric_m"]
                    if isinstance(m_val, list) and len(m_val) > 0:
                        baseline_forecast_res["m"] = m_val[0]
                    else:
                        baseline_forecast_res["m"] = m_val
                else:
                    try:
                        project_root = get_project_root(cfg_path)
                        b_cfg_path = find_config_path(project_root, baseline_variant)
                        if b_cfg_path is not None:
                            cfg = get_config(str(b_cfg_path))
                            m_seconds = cfg.model.forecast.m
                            sampling_freq = cfg.data.sampling_frequency
                            baseline_forecast_res["m"] = int(m_seconds * sampling_freq)
                    except Exception:
                        baseline_forecast_res["m"] = 0

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
                "Xp",
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

            t_abs_margined = None
            if meta_time_margined and len(meta_time_margined) > trial_idx:
                t_margined_val = meta_time_margined[trial_idx]
                if t_margined_val is not None:
                    t_full = np.array(t_margined_val)
                    if t_full.ndim > 0 and len(t_full) >= len(y_concat):
                        t_abs_margined = t_full[: len(y_concat)]

            if t_abs_margined is None:
                t_abs_margined = np.linspace(
                    0.0, len(y_concat) / SAMPLING_FREQ, len(y_concat)
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

            baseline_yp_c_f = None
            if baseline_forecast_res and "Y_future_pred" in baseline_forecast_res:
                by_fp = np.array(baseline_forecast_res["Y_future_pred"])
                if by_fp.ndim == 1:
                    baseline_yp_c_f = by_fp
                elif by_fp.ndim == 2 and c < by_fp.shape[1]:
                    baseline_yp_c_f = by_fp[:, c]
                else:
                    baseline_yp_c_f = by_fp.flatten()

            cfg = get_config(str(cfg_path))
            fs = getattr(cfg.data, "sampling_frequency", SAMPLING_FREQ)

            y_future_true_arr = np.array(y_future_true)
            y_future_pred_arr = np.array(y_future_pred)

            y_true_ch = (
                get_channel(y_future_true_arr, c, t_abs_margined[:m])
                if y_future_true_arr.ndim == 2
                else y_future_true_arr.flatten()
            )
            y_pred_ch = (
                get_channel(y_future_pred_arr, c, t_abs_margined[:m])
                if y_future_pred_arr.ndim == 2
                else y_future_pred_arr.flatten()
            )

            T = len(y_concat)
            Tpast = max(0, T - m)
            t_future = t_abs_margined[Tpast:T]

            baseline_r_f = None
            if baseline_yp_c_f is not None:
                try:
                    baseline_r_f = np.corrcoef(
                        y_true_ch.flatten(), baseline_yp_c_f.flatten()
                    )[0, 1]
                except Exception:
                    baseline_r_f = np.nan

            render_y_forecast_plot(
                y_concat,
                y_future_true,
                y_future_pred,
                t_abs_margined,
                m,
                c,
                selected_name,
                r_fore_ch,
                baseline_yp_c=baseline_yp_c_f,
                baseline_name=selected_baseline_name,
                model_name=model_label,
                baseline_r=baseline_r_f,
            )

            # Y forecast analysis pipeline (shared)
            render_analysis(
                y_true_ch,
                y_pred_ch,
                t_future,
                selected_name,
                r_fore_ch,
                sampling_rate=fs,
                unit="µV",
                baseline_pred_c=baseline_yp_c_f,
                baseline_r=baseline_r_f,
                baseline_name=selected_baseline_name,
                model_name=model_label,
                diagnostics_label="Residual Diagnostics & Normality Tests (Forecast)",
            )

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

                    z_ft_c = (
                        z_future_true.squeeze()
                        if nz_chan == 1
                        else z_future_true[:, z_c]
                    )
                    z_fp_c = (
                        z_future_pred.squeeze()
                        if nz_chan == 1
                        else z_future_pred[:, z_c]
                    )

                    r_fore_z_ch = np.nan
                    if r_fore_list_z is not None:
                        if (
                            isinstance(r_fore_list_z, (list, tuple, np.ndarray))
                            and len(r_fore_list_z) > z_c
                        ):
                            r_fore_z_ch = r_fore_list_z[z_c]

                    zp_1_f, zp_2_f, r_zp1_f, r_zp2_f = None, None, None, None
                    try:
                        n1 = getattr(get_config(str(cfg_path)).model, "n1", 0)
                        if (
                            "Xp" in split_res
                            and split_res["Xp"] is not None
                            and "B_z" in split_res
                            and split_res["B_z"] is not None
                            and n1 > 0
                        ):
                            Xp_trial = split_res["Xp"][trial_idx]
                            B_z = split_res["B_z"]
                            d_z = split_res.get("d_z")
                            Xp_future = (
                                np.array(Xp_trial)[-m:]
                                if Xp_trial is not None
                                else None
                            )
                            if Xp_future is not None and B_z is not None:
                                zp_1_f, zp_2_f, r_zp1_f, r_zp2_f = (
                                    _compute_zp_components(
                                        z_ft_c, Xp_future, B_z, d_z, n1, z_c
                                    )
                                )
                    except Exception:
                        pass

                    # Extract baseline Z forecast channel
                    baseline_zp_c_f = None
                    baseline_r_z_f = None
                    if (
                        baseline_forecast_res
                        and "Z_future_pred" in baseline_forecast_res
                    ):
                        bz_fp = np.array(baseline_forecast_res["Z_future_pred"])
                        if bz_fp.ndim == 1:
                            baseline_zp_c_f = bz_fp
                        elif bz_fp.ndim == 2 and z_c < bz_fp.shape[1]:
                            baseline_zp_c_f = bz_fp[:, z_c]
                        else:
                            baseline_zp_c_f = bz_fp.flatten()
                        if baseline_zp_c_f is not None:
                            try:
                                baseline_r_z_f = np.corrcoef(
                                    z_ft_c.flatten(), baseline_zp_c_f.flatten()
                                )[0, 1]
                            except Exception:
                                baseline_r_z_f = np.nan

                    st.markdown("#### Time Series: True Future vs Forecast")
                    render_z_forecast_plot(
                        z_concat,
                        z_future_true,
                        z_future_pred,
                        t_abs_margined,
                        m,
                        z_c,
                        selected_z_name,
                        r_fore_z_ch,
                        zp_1=zp_1_f,
                        zp_2=zp_2_f,
                        r_zp1=r_zp1_f,
                        r_zp2=r_zp2_f,
                        baseline_zp_c=baseline_zp_c_f,
                        baseline_name=selected_baseline_name,
                        model_name=model_label,
                        baseline_r=baseline_r_z_f,
                    )

                    # Z forecast analysis pipeline (shared, with rescale)
                    Tpast_z = max(0, len(t_abs_margined) - m)
                    t_future_z = t_abs_margined[Tpast_z : Tpast_z + len(z_ft_c)]

                    render_analysis(
                        z_ft_c,
                        z_fp_c,
                        t_future_z,
                        selected_z_name,
                        r_fore_z_ch,
                        sampling_rate=fs,
                        baseline_pred_c=baseline_zp_c_f,
                        baseline_r=baseline_r_z_f,
                        baseline_name=selected_baseline_name,
                        model_name=model_label,
                        rescale=True,
                        show_psd=False,
                        diagnostics_label="Residual Diagnostics & Normality Tests (Forecast)",
                    )

                except Exception:
                    pass

            x_future_pred = f_res.get("X_future_pred")

            if x_future_pred is not None:
                try:
                    x_future_pred = np.array(x_future_pred)

                    cfg = get_config(str(cfg_path))
                    n1 = getattr(cfg.model, "n1", 0)

                    Xp = split_res.get("Xp", [])
                    if Xp and len(Xp) > trial_idx and Xp[trial_idx] is not None:
                        x_p_trial = np.array(Xp[trial_idx])

                        Tpast = len(y_concat) - m
                        x_history = x_p_trial[:Tpast]

                        st.markdown("---")
                        st.subheader("Latent States Forecast")
                        render_x_forecast_plot(
                            x_history, x_future_pred, t_abs_margined, m, n1
                        )

                except Exception as e:
                    st.warning(f"Could not render latent forecast plot: {e}")

    except Exception:
        pass
