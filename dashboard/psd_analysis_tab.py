import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.plots import (
    plot_psd_heatmap,
    plot_average_psd,
    plot_average_psd_dbs_comparison,
)
from dashboard.utils import get_channel_lists, get_trial_metadata
from dashboard.backbone import format_trial_metadata, update_fig_title


def compute_per_trial_psd_means(df, channel):
    means = []
    if df.is_empty():
        return means

    col = f"{channel}_psd_values"
    for trial_psds in df[col].to_list():
        arr = np.array(trial_psds)
        if arr.ndim == 1:
            means.append(arr)
        elif arr.ndim >= 2 and arr.size > 0:
            means.append(arr.mean(axis=0))
    return means


def get_psd_time_info(trial_data, lfp_channels):
    tvec = np.array(trial_data["time_original"][0])
    onset = float(trial_data["onset"][0])
    n_epochs = len(trial_data[f"{lfp_channels[0]}_psd_values"][0])
    times_abs = np.linspace(tvec.min(), tvec.max(), n_epochs)
    return times_abs, onset, n_epochs


def render_psd_heatmap(
    trial_data, channel, channel_type, times_abs, onset, show_rel_axis, metadata_str
):

    freqs = np.array(trial_data[f"{channel}_psd_freq"][0])
    psd_values = np.array(trial_data[f"{channel}_psd_values"][0])

    fig = plot_psd_heatmap(
        freqs,
        psd_values,
        title="",
        times_abs=times_abs,
        add_rel_axis=show_rel_axis,
        rel_offset=onset,
    )

    fig = update_fig_title(fig, [f"{channel} PSD Heatmap", metadata_str])
    st.plotly_chart(fig, use_container_width=True)


def render_average_psd_trial_level(trial_data, channels, channel_type, metadata_str):

    if not channels:
        return

    psd_data = {}
    freqs = None

    for ch in channels:
        if freqs is None and not trial_data.is_empty():
            freqs = np.array(trial_data[f"{ch}_psd_freq"][0])

        psd_values = np.array(trial_data[f"{ch}_psd_values"][0])

        psd_values = np.vstack([np.array(x) for x in psd_values])

        mean_psd = psd_values.mean(axis=0)
        mean_psd = np.maximum(mean_psd, 1e-20)

        psd_data[ch] = {"on": np.array([mean_psd]), "off": np.array([])}

    if freqs is not None:
        fig = plot_average_psd(freqs, psd_data, title="")

        title_parts = [f"Trial-level Average {channel_type} PSD", metadata_str]
        fig = update_fig_title(fig, title_parts)
        st.plotly_chart(fig, use_container_width=True)


def render_average_psd_session_level(channels, channel_type):
    if not channels:
        return

    from utils.data_loader import load_participant_session_data

    participant_id = st.session_state.get("participant_id")
    session = st.session_state.get("session")

    if not participant_id or not session:
        st.warning("Session information not available")
        return

    try:
        session_data = load_participant_session_data(participant_id, session)
    except Exception as e:
        st.error(f"Could not load session data: {e}")
        return

    if session_data.is_empty():
        st.info("No session data available")
        return

    dbs_on_data = session_data.filter(pl.col("stim") == "on")
    dbs_off_data = session_data.filter(pl.col("stim") == "off")

    psd_data = {}
    freqs = None

    for ch in channels:
        if freqs is None and not session_data.is_empty():
            freqs = np.array(session_data[f"{ch}_psd_freq"][0])

        on_means = compute_per_trial_psd_means(dbs_on_data, ch)
        off_means = compute_per_trial_psd_means(dbs_off_data, ch)

        psds_on = np.vstack(on_means) if len(on_means) > 0 else np.array([])
        psds_off = np.vstack(off_means) if len(off_means) > 0 else np.array([])

        psd_data[ch] = {"on": psds_on, "off": psds_off}

    if freqs is not None:
        fig_dbs = plot_average_psd_dbs_comparison(freqs, psd_data, title="")

        title_parts_dbs = [
            f"Session-level Average {channel_type} PSD • DBS ON vs OFF",
            f"Participant {participant_id} • Session {session}",
        ]
        fig_dbs = update_fig_title(fig_dbs, title_parts_dbs)
        st.plotly_chart(fig_dbs, use_container_width=True)


def render_psd_heatmap_tab(trial_data, lfp_channels, ecog_channels, metadata_str):
    times_abs, onset, n_epochs = get_psd_time_info(trial_data, lfp_channels)

    show_rel_axis = True

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### LFP PSD Heatmap")
        selected_lfp = st.selectbox(
            "Select LFP Channel", lfp_channels, key="lfp_psd_heatmap"
        )
        if selected_lfp:
            render_psd_heatmap(
                trial_data,
                selected_lfp,
                "LFP",
                times_abs,
                onset,
                show_rel_axis,
                metadata_str,
            )

    with col2:
        st.markdown("#### ECoG PSD Heatmap")
        selected_ecog = st.selectbox(
            "Select ECoG Channel", ecog_channels, key="ecog_psd_heatmap"
        )
        if selected_ecog:
            render_psd_heatmap(
                trial_data,
                selected_ecog,
                "ECoG",
                times_abs,
                onset,
                show_rel_axis,
                metadata_str,
            )


def render_average_psd_tab(
    trial_data, block_data, lfp_channels, ecog_channels, metadata_str
):
    st.markdown("### Trial-level Average PSD")
    st.markdown("Average PSD for the selected trial across time epochs")

    col1_trial, col2_trial = st.columns(2)

    with col1_trial:
        st.markdown("#### LFP")
        selected_lfp_trial = st.multiselect(
            "Select LFP Channels", lfp_channels, key="lfp_psd_avg_trial"
        )
        if selected_lfp_trial:
            render_average_psd_trial_level(
                trial_data, selected_lfp_trial, "LFP", metadata_str
            )

    with col2_trial:
        st.markdown("#### ECoG")
        selected_ecog_trial = st.multiselect(
            "Select ECoG Channels", ecog_channels, key="ecog_psd_avg_trial"
        )
        if selected_ecog_trial:
            render_average_psd_trial_level(
                trial_data, selected_ecog_trial, "ECoG", metadata_str
            )

    st.markdown("---")

    st.markdown("### Session-level Average PSD")
    st.markdown("Average PSD across all blocks in the session, comparing DBS ON vs OFF")

    col1_session, col2_session = st.columns(2)

    with col1_session:
        st.markdown("#### LFP")
        selected_lfp_session = st.multiselect(
            "Select LFP Channels", lfp_channels, key="lfp_psd_avg_session"
        )
        if selected_lfp_session:
            render_average_psd_session_level(selected_lfp_session, "LFP")

    with col2_session:
        st.markdown("#### ECoG")
        selected_ecog_session = st.multiselect(
            "Select ECoG Channels", ecog_channels, key="ecog_psd_avg_session"
        )
        if selected_ecog_session:
            render_average_psd_session_level(selected_ecog_session, "ECoG")


def psd_analysis_tab(block_data, lfp_channels, ecog_channels):
    st.header("Power Spectral Density (PSD) Analysis")

    trials_in_block = sorted(block_data["trial"].unique().to_list())
    selected_trial = st.selectbox(
        "Select a Trial to Render",
        options=trials_in_block,
        key="psd_trial_selector",
    )

    trial_data = block_data.filter(pl.col("trial") == selected_trial)

    if trial_data is None or trial_data.is_empty():
        st.info("Select a block and trial to view PSD data.")
        return

    meta = get_trial_metadata(trial_data)

    metadata_str = format_trial_metadata(
        meta.get("participant_id"),
        meta.get("session"),
        meta.get("block"),
        meta.get("trial"),
        stim=meta.get("stim"),
    )

    st.subheader(metadata_str)

    psd_tab, avg_psd_tab = st.tabs(["PSD Heatmap", "Average PSD"])

    with psd_tab:
        render_psd_heatmap_tab(trial_data, lfp_channels, ecog_channels, metadata_str)

    with avg_psd_tab:
        render_average_psd_tab(
            trial_data, block_data, lfp_channels, ecog_channels, metadata_str
        )
