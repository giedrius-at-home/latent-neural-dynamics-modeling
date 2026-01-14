import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, Dict, Any, List

from dashboard.backbone import PALETTE
from dashboard.subtabs.helpers import (
    get_trial_time_axis,
    compute_forecast_for_trial,
)
from utils.config import get_config
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

    m_seconds = m_samples / SAMPLING_FREQ
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

    mean_true = np.mean(z_ft_c)
    mean_pred = np.mean(z_fp_c)
    std_true = np.std(z_ft_c)
    std_pred = np.std(z_fp_c)
    if std_pred > 0:
        scale_factor = std_true / std_pred
        z_fp_c = (z_fp_c - mean_pred) * scale_factor + mean_true
    else:
        scale_factor = 1.0
        z_fp_c = z_fp_c - mean_pred + mean_true

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
            name=f"Forecast (scaled ×{scale_factor:.2f})",
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

    m_seconds = m_samples / SAMPLING_FREQ
    fig.update_layout(
        title=f"Z Forecast (horizon={m_seconds:.2f}s) — {channel_name} (r={r_fore_z_ch:.3f})",
        xaxis_title="Time (s)",
        yaxis_title="Value",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_frequency_analysis(
    y_future_true: np.ndarray,
    y_future_pred: np.ndarray,
    channel_name: str,
    sampling_freq: float = SAMPLING_FREQ,
):

    st.markdown(f"### Frequency Analysis — {channel_name}")
    st.markdown("Compare frequency content of true vs predicted signals")

    freqs_true, psd_true = compute_power_spectrum(y_future_true, sampling_freq)
    freqs_pred, psd_pred = compute_power_spectrum(y_future_pred, sampling_freq)

    psd_true_1d = psd_true.flatten() if psd_true.ndim > 1 else psd_true
    psd_pred_1d = psd_pred.flatten() if psd_pred.ndim > 1 else psd_pred

    from utils.stats import (
        compare_band_power,
        CLINICAL_FREQUENCY_BANDS,
    )

    band_comparison = compare_band_power(
        y_future_true, y_future_pred, sampling_freq, CLINICAL_FREQUENCY_BANDS
    )

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "PSD Overlay",
            "Band Power Comparison",
        ],
        horizontal_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=freqs_true,
            y=psd_true_1d,
            mode="lines",
            name="True",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=freqs_pred,
            y=psd_pred_1d,
            mode="lines",
            name="Predicted",
            line=dict(color="red", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    band_names = list(band_comparison.keys())
    true_powers = [band_comparison[b]["true"] for b in band_names]
    pred_powers = [band_comparison[b]["pred"] for b in band_names]

    fig.add_trace(
        go.Bar(
            x=band_names,
            y=true_powers,
            name="True",
            marker_color="blue",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=band_names,
            y=pred_powers,
            name="Predicted",
            marker_color="red",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="PSD", type="log", row=1, col=1)

    fig.update_xaxes(title_text="Frequency Band", row=1, col=2)
    fig.update_yaxes(title_text="Power", row=1, col=2)

    fig.update_layout(
        height=400, showlegend=True, title_text=f"Frequency Analysis — {channel_name}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Band Power Statistics")

    import pandas as pd

    band_stats = []
    for band_name, stats in band_comparison.items():
        freq_range = stats["freq_range"]
        band_stats.append(
            {
                "Band": f"{band_name} ({freq_range[0]}-{freq_range[1]} Hz)",
                "True Power": f"{stats['true']:.2e}",
                "Pred Power": f"{stats['pred']:.2e}",
                "Error": f"{stats['error']:.2e}",
                "Ratio": (
                    f"{stats['ratio']:.3f}" if not np.isnan(stats["ratio"]) else "N/A"
                ),
            }
        )

    df_bands = pd.DataFrame(band_stats)
    st.dataframe(df_bands, use_container_width=True, hide_index=True)


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

                y_future_true_arr = np.array(y_future_true)
                y_future_pred_arr = np.array(y_future_pred)

                if y_future_true_arr.ndim == 1:
                    y_true_ch = y_future_true_arr
                elif y_future_true_arr.ndim == 2 and c < y_future_true_arr.shape[1]:
                    y_true_ch = y_future_true_arr[:, c]
                else:
                    y_true_ch = y_future_true_arr.flatten()

                if y_future_pred_arr.ndim == 1:
                    y_pred_ch = y_future_pred_arr
                elif y_future_pred_arr.ndim == 2 and c < y_future_pred_arr.shape[1]:
                    y_pred_ch = y_future_pred_arr[:, c]
                else:
                    y_pred_ch = y_future_pred_arr.flatten()

                if y_true_ch.ndim == 0 or y_pred_ch.ndim == 0:
                    st.warning("Invalid channel data for frequency analysis")
                else:
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
