import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
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
from utils.stats import (
    compute_power_spectrum,
    find_dominant_frequencies,
    spectral_correlation,
)

SAMPLING_FREQ = 60


def _block_trial_indices_by_block(split_res: Dict[str, Any]) -> Dict[Tuple, List[int]]:
    blk_list = split_res.get("block", [])
    pid_list = split_res.get("participant_id", [])
    ses_list = split_res.get("session", [])
    n_trials = len(split_res.get("Y", []))
    if not blk_list and n_trials > 0:
        return {("all",): list(range(n_trials))}
    block_to_trials = {}
    for t in range(n_trials):
        pid = pid_list[t] if pid_list and t < len(pid_list) else None
        ses = ses_list[t] if ses_list and t < len(ses_list) else None
        blk = blk_list[t] if blk_list and t < len(blk_list) else t
        key = (pid, ses, blk)
        block_to_trials.setdefault(key, []).append(t)
    return block_to_trials


def _get_or_compute_forecast_for_trial(
    split_res: Dict[str, Any],
    trial_idx: int,
    cfg_path: Path,
    run_ts: str,
    project_root: Path,
    results_root: Path,
    split_name: str,
    baseline_res: Optional[Dict],
    baseline_variant: Optional[str],
) -> Tuple[Optional[Dict], Optional[Dict]]:
    Y_true = split_res.get("Y", [])
    Z_list = split_res.get("Z")
    chunk_margins = split_res.get("chunk_margin")
    if trial_idx >= len(Y_true):
        return None, None

    main_f = None
    if "Y_future_pred" in split_res and split_res["Y_future_pred"] is not None:
        if (
            trial_idx < len(split_res["Y_future_pred"])
            and split_res["Y_future_pred"][trial_idx] is not None
        ):
            m_val = split_res.get("metric_m")
            m = (
                int(m_val[0] if isinstance(m_val, list) and m_val else m_val)
                if m_val
                else 0
            )
            if not m and cfg_path:
                cfg = get_config(str(cfg_path))
                m = int(
                    getattr(cfg.model.forecast, "m", 2.0)
                    * getattr(cfg.data, "sampling_frequency", 60)
                )
            main_f = {
                "m": m,
                "Y_future_true": (
                    split_res["Y_future_true"][trial_idx]
                    if split_res.get("Y_future_true")
                    else None
                ),
                "Y_future_pred": split_res["Y_future_pred"][trial_idx],
                "Y_concat_for_plot": (
                    split_res["Y_concat_for_plot"][trial_idx]
                    if split_res.get("Y_concat_for_plot")
                    else None
                ),
            }
    if (
        main_f is None
        and "trial_forecasts" in split_res
        and trial_idx in split_res["trial_forecasts"]
    ):
        main_f = split_res["trial_forecasts"][trial_idx]
    if main_f is None:
        y_trial = np.array(Y_true[trial_idx])
        z_trial = (
            np.array(Z_list[trial_idx])
            if Z_list and trial_idx < len(Z_list) and Z_list[trial_idx] is not None
            else None
        )
        chunk_margin = (
            chunk_margins[trial_idx]
            if isinstance(chunk_margins, (list, np.ndarray))
            and chunk_margins is not None
            and trial_idx < len(chunk_margins)
            else None
        )
        if chunk_margin is None and isinstance(
            split_res.get("chunk_margin"), (list, np.ndarray)
        ):
            chunk_margin = (
                split_res["chunk_margin"][trial_idx]
                if trial_idx < len(split_res["chunk_margin"])
                else None
            )
        try:
            main_f = compute_forecast_for_trial(
                str(cfg_path), run_ts, y_trial, z_trial, chunk_margin
            )
            if "trial_forecasts" not in split_res:
                split_res["trial_forecasts"] = {}
            split_res["trial_forecasts"][trial_idx] = main_f
        except Exception:
            return None, None

    base_f = None
    if baseline_res is not None and baseline_variant:
        if (
            "Y_future_pred" in baseline_res
            and baseline_res["Y_future_pred"] is not None
            and trial_idx < len(baseline_res["Y_future_pred"])
            and baseline_res["Y_future_pred"][trial_idx] is not None
        ):
            m_val = baseline_res.get("metric_m")
            m = (
                int(m_val[0] if isinstance(m_val, list) and m_val else m_val)
                if m_val
                else main_f.get("m", 0)
            )
            base_f = {
                "m": m,
                "Y_future_true": (
                    baseline_res["Y_future_true"][trial_idx]
                    if baseline_res.get("Y_future_true")
                    else None
                ),
                "Y_future_pred": baseline_res["Y_future_pred"][trial_idx],
            }
        elif main_f and main_f.get("m"):
            baseline_cfg = find_config_path(project_root, baseline_variant)
            baseline_ts = st.session_state.get("fore_baseline_ts")
            if baseline_cfg and baseline_ts:
                y_trial = np.array(Y_true[trial_idx])
                z_trial = (
                    np.array(Z_list[trial_idx])
                    if Z_list
                    and trial_idx < len(Z_list)
                    and Z_list[trial_idx] is not None
                    else None
                )
                chunk_margin = (
                    chunk_margins[trial_idx]
                    if isinstance(chunk_margins, (list, np.ndarray))
                    and chunk_margins is not None
                    and trial_idx < len(chunk_margins)
                    else None
                )
                try:
                    base_f = compute_forecast_for_trial(
                        str(baseline_cfg), baseline_ts, y_trial, z_trial, chunk_margin
                    )
                except Exception:
                    pass
    return main_f, base_f


def render_block_forecast_view(
    split_res: Dict[str, Any],
    cfg_path: Path,
    run_ts: str,
    baseline_res: Optional[Dict[str, Any]],
    baseline_variant: Optional[str],
    selected_baseline_name: str,
    model_label: str,
    split_name: str,
):
    n_trials = len(split_res.get("Y", []))
    if n_trials == 0:
        st.info("No trials in this split.")
        return

    block_to_trials = _block_trial_indices_by_block(split_res)
    if not block_to_trials:
        block_to_trials = {("all",): list(range(n_trials))}
    project_root = get_project_root(cfg_path)
    results_root = project_root / "results"
    try:
        cfg = get_config(str(cfg_path))
    except Exception:
        cfg = None
    fs = (
        getattr(cfg.data, "sampling_frequency", SAMPLING_FREQ) if cfg else SAMPLING_FREQ
    )

    block_options = []
    for key, indices in block_to_trials.items():
        if key == ("all",):
            label = f"All trials ({len(indices)})"
        else:
            _, ses, blk = (
                key if len(key) == 3 else (None, None, key[0] if len(key) == 1 else key)
            )
            label = (
                f"Block {blk}"
                + (f" (session {ses})" if ses is not None else "")
                + f" — {len(indices)} trials"
            )
        block_options.append((key, label, indices))

    st.markdown("#### Block view")
    selected_block_idx = st.selectbox(
        "Block",
        options=list(range(len(block_options))),
        format_func=lambda i: block_options[i][1],
        key="block_forecast_block_select",
    )
    _, _, block_trial_indices = block_options[selected_block_idx]

    with st.spinner("Loading or computing forecasts for block..."):
        main_list = []
        base_list = []
        m_used = None
        for t in block_trial_indices:
            main_f, base_f = _get_or_compute_forecast_for_trial(
                split_res,
                t,
                cfg_path,
                run_ts,
                project_root,
                results_root,
                split_name,
                baseline_res,
                baseline_variant,
            )
            main_list.append(main_f)
            base_list.append(base_f)
            if main_f and main_f.get("m"):
                m_used = main_f["m"]

        if not main_list or not any(main_list):
            st.warning("No forecast data available for trials in this block.")
            return
        if m_used is None and cfg:
            m_used = int(getattr(cfg.model.forecast, "m", 2.0) * fs)
        if m_used is None:
            m_used = int(2.0 * fs)

    n_chan = None
    for mf in main_list:
        if mf and mf.get("Y_future_true") is not None:
            yt = np.array(mf["Y_future_true"])
            n_chan = yt.shape[1] if yt.ndim == 2 else 1
            break
    if n_chan is None:
        st.warning("Could not infer number of channels.")
        return

    chan_names = split_res.get("input_channels", [])
    if not chan_names or len(chan_names) != n_chan:
        chan_names = [f"ch{i}" for i in range(n_chan)]

    t_future_full = np.arange(m_used) / fs
    n_rows = n_chan
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=chan_names,
        vertical_spacing=0.06,
        shared_xaxes=True,
    )

    for ch in range(n_chan):
        for tri_idx, (main_f, base_f) in enumerate(zip(main_list, base_list)):
            if main_f is None or main_f.get("Y_future_true") is None:
                continue
            y_true = np.array(main_f["Y_future_true"])
            y_pred = np.array(main_f["Y_future_pred"])
            y_true_c = y_true.squeeze() if n_chan == 1 else y_true[:, ch]
            y_pred_c = y_pred.squeeze() if n_chan == 1 else y_pred[:, ch]
            show_legend_true = ch == 0 and tri_idx == 0
            show_legend_main = ch == 0 and tri_idx == 0
            show_legend_base = ch == 0 and tri_idx == 0
            fig.add_trace(
                go.Scatter(
                    x=t_future_full,
                    y=y_true_c,
                    name="True",
                    legendgroup="True",
                    showlegend=show_legend_true,
                    mode="lines",
                    line=dict(color=PLOT_COLOR.stim_off, width=0.8),
                ),
                row=ch + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_future_full,
                    y=y_pred_c,
                    name=model_label,
                    legendgroup=model_label,
                    showlegend=show_legend_main,
                    mode="lines",
                    line=dict(color=PLOT_COLOR.stim_on, width=0.8, dash="dot"),
                ),
                row=ch + 1,
                col=1,
            )
            if base_f is not None and base_f.get("Y_future_pred") is not None:
                y_base = np.array(base_f["Y_future_pred"])
                y_base_c = y_base.squeeze() if n_chan == 1 else y_base[:, ch]
                fig.add_trace(
                    go.Scatter(
                        x=t_future_full,
                        y=y_base_c,
                        name=selected_baseline_name,
                        legendgroup=selected_baseline_name,
                        showlegend=show_legend_base,
                        mode="lines",
                        line=dict(color=BASELINE_COLOR, width=0.8, dash="dash"),
                    ),
                    row=ch + 1,
                    col=1,
                )

    fig.update_layout(
        height=max(220 * n_rows, 300),
        margin=dict(t=40, b=40, l=50, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Amplitude (µV)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)
    st.plotly_chart(fig, use_container_width=True, key="block_forecast_fig")


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
        try:
            st.markdown("---")
            render_block_forecast_view(
                split_res,
                cfg_path,
                run_ts,
                baseline_res,
                baseline_variant,
                selected_baseline_name,
                model_label,
                split_name,
            )
        except Exception as e:
            st.warning(f"Block view: {e}")
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

            st.markdown("---")
            try:
                render_block_forecast_view(
                    split_res,
                    cfg_path,
                    run_ts,
                    baseline_res,
                    baseline_variant,
                    selected_baseline_name,
                    model_label,
                    split_name,
                )
            except Exception as e:
                st.warning(f"Block view: {e}")

    except Exception:
        pass
