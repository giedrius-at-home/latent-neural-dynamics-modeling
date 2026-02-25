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
from utils.stats import compute_psd_dbs_stats
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
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Trial-level average PSD — {metadata_str} — Channels: {', '.join(channels)}"
        )


def render_average_psd_session_level(channels, channel_type):
    if not channels:
        return

    from utils.data_loader import load_participant_session_data

    participant_id = st.session_state.get("participant_id")
    session = st.session_state.get("session")

    if not participant_id or not session:
        return

    selected_dataset = st.session_state.get("selected_dataset")

    cols_to_load = (
        [f"{ch}_psd_values" for ch in channels]
        + [f"{ch}_psd_freq" for ch in channels]
        + ["stim"]
    )
    try:
        session_data = load_participant_session_data(
            participant_id, session, selected_dataset, columns=cols_to_load
        )
    except Exception:
        return

    if session_data.is_empty():
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
        st.plotly_chart(fig_dbs, use_container_width=True)
        st.caption(
            f"Session-level average PSD comparison (DBS ON vs OFF) — {participant_id}, Session: {session} — Channels: {', '.join(channels)}"
        )

        render_psd_statistical_comparison(freqs, psd_data, participant_id, session)


def render_psd_statistical_comparison(freqs, psd_data, participant_id, session):
    """Render DBS ON vs OFF PSD statistical comparison per channel."""
    rows = []
    for ch, data in psd_data.items():
        has_on = "on" in data and data["on"].size > 0
        has_off = "off" in data and data["off"].size > 0
        if not (has_on and has_off):
            continue

        result = compute_psd_dbs_stats(freqs, data["on"], data["off"])
        rows.append({"Channel": ch, **result})

    if not rows:
        return

    import polars as pl

    df = pl.DataFrame(rows).select(
        "Channel",
        pl.col("n_on").alias("N (ON)"),
        pl.col("n_off").alias("N (OFF)"),
        pl.col("mean_on").round(2).alias("Mean ON (dB)"),
        pl.col("mean_off").round(2).alias("Mean OFF (dB)"),
        pl.col("delta_db").round(2).alias("Δ (dB)"),
        pl.col("U").round(1).alias("U"),
        pl.col("p").alias("p-value"),
        pl.col("effect_size_r").round(3).alias("Effect Size (r)"),
    )

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        f"Mann–Whitney U test — DBS ON vs OFF integrated PSD power — "
        f"{participant_id}, Session: {session}"
    )

    # Grouped bar chart
    fig = go.Figure()
    channels = [r["Channel"] for r in rows]
    means_on = [r["mean_on"] for r in rows]
    means_off = [r["mean_off"] for r in rows]

    fig.add_trace(
        go.Bar(
            name="DBS ON",
            x=channels,
            y=means_on,
            marker_color="#ff0035",
        )
    )
    fig.add_trace(
        go.Bar(
            name="DBS OFF",
            x=channels,
            y=means_off,
            marker_color="#59546c",
        )
    )

    # Add significance annotations
    for i, r in enumerate(rows):
        p = r.get("p", np.nan)
        if np.isnan(p):
            continue
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            continue
        y_max = max(r["mean_on"], r["mean_off"])
        fig.add_annotation(
            x=channels[i],
            y=y_max + 0.5,
            text=star,
            showarrow=False,
            font=dict(size=14),
        )

    fig.update_layout(
        barmode="group",
        xaxis_title="Channel",
        yaxis_title="Mean Integrated Power (dB)",
        template="plotly_white",
        showlegend=True,
        margin=dict(l=60, r=40, t=30, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Mean integrated PSD power per channel — " f"* p<0.05, ** p<0.01, *** p<0.001"
    )


def render_average_psd_participant_level(channels, channel_type):
    if not channels:
        return

    from utils.data_loader import load_participant_data
    from dashboard.backbone import PARTICIPANT_COLORS, PLOT_STYLE

    participant_id = st.session_state.get("participant_id")
    selected_dataset = st.session_state.get("selected_dataset")

    if not participant_id:
        st.info("Please select a participant in the sidebar.")
        return

    cols_to_load = (
        [f"{ch}_psd_values" for ch in channels]
        + [f"{ch}_psd_freq" for ch in channels]
        + ["session", "stim"]
    )
    try:
        participant_data = load_participant_data(
            participant_id, selected_dataset, columns=cols_to_load
        )
    except Exception as e:
        st.error(f"Could not load participant data: {str(e)}")
        return

    if participant_data.is_empty():
        st.info("No data available for this participant.")
        return

    first_channel = channels[0]
    psd_col = f"{first_channel}_psd_values"
    freq_col = f"{first_channel}_psd_freq"

    if (
        psd_col not in participant_data.columns
        or freq_col not in participant_data.columns
    ):
        st.warning(
            f"PSD data not found for {first_channel}. Available columns: {', '.join([c for c in participant_data.columns if 'psd' in c.lower()])}"
        )
        return

    sessions = sorted(participant_data["session"].unique().to_list())

    st.info(
        f"Showing PSD comparison across {len(sessions)} session(s) for {participant_id}"
    )

    session_colors = px.colors.qualitative.Plotly

    for ch in channels:
        psd_col = f"{ch}_psd_values"
        freq_col = f"{ch}_psd_freq"

        if (
            psd_col not in participant_data.columns
            or freq_col not in participant_data.columns
        ):
            st.warning(f"PSD data not found for channel {ch}")
            continue

        freqs = None
        fig = go.Figure()

        for idx, session in enumerate(sessions):
            session_data = participant_data.filter(pl.col("session") == session)
            if session_data.is_empty():
                continue

            session_color = session_colors[idx % len(session_colors)]

            if freqs is None:
                try:
                    freqs = np.array(session_data[freq_col][0])
                except Exception as e:
                    st.warning(f"Could not extract frequencies for {ch}: {str(e)}")
                    continue

            dbs_on_data = session_data.filter(pl.col("stim") == "on")
            dbs_off_data = session_data.filter(pl.col("stim") == "off")

            on_means = compute_per_trial_psd_means(dbs_on_data, ch)
            off_means = compute_per_trial_psd_means(dbs_off_data, ch)

            if len(on_means) > 0:
                psds_on = np.vstack(on_means)
                mean_psd_on = 10 * np.log10(np.mean(psds_on, axis=0)) + 120
                fig.add_trace(
                    go.Scatter(
                        x=freqs,
                        y=mean_psd_on,
                        mode="lines",
                        name=f"Session {session} ON",
                        line=dict(color=session_color, width=1.2),
                    )
                )

            if len(off_means) > 0:
                psds_off = np.vstack(off_means)
                mean_psd_off = 10 * np.log10(np.mean(psds_off, axis=0)) + 120
                fig.add_trace(
                    go.Scatter(
                        x=freqs,
                        y=mean_psd_off,
                        mode="lines",
                        name=f"Session {session} OFF",
                        line=dict(color=session_color, width=1.2, dash="dot"),
                    )
                )

        if freqs is not None and len(fig.data) > 0:
            fig.update_layout(
                title=f"{ch} - {participant_id} (All Sessions)",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Power/Frequency (dB/Hz)",
                template="plotly_white",
                font=dict(family=PLOT_STYLE.font_family, size=12, color="#0e131f"),
                legend=dict(font=dict(size=10, family=PLOT_STYLE.font_family)),
                showlegend=True,
                margin=dict(l=60, r=80, t=60, b=60),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"PSD comparison across sessions for {participant_id} — Channel: {ch}"
            )
        else:
            st.warning(f"No PSD data could be plotted for channel {ch}")


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

    st.markdown("---")

    st.markdown("### Participant-level Average PSD")

    col1_part, col2_part = st.columns(2)

    with col1_part:
        st.markdown("#### LFP")
        selected_lfp_part = st.multiselect(
            "Select LFP Channels", lfp_channels, key="lfp_psd_avg_participant"
        )
        if selected_lfp_part:
            render_average_psd_participant_level(selected_lfp_part, "LFP")

    with col2_part:
        st.markdown("#### ECoG")
        selected_ecog_part = st.multiselect(
            "Select ECoG Channels", ecog_channels, key="ecog_psd_avg_participant"
        )
        if selected_ecog_part:
            render_average_psd_participant_level(selected_ecog_part, "ECoG")


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

    psd_tab, avg_psd_tab, multi_psd_tab = st.tabs(
        ["PSD Heatmap", "Average PSD", "Multi-Participant PSD"]
    )

    with psd_tab:
        render_psd_heatmap_tab(trial_data, lfp_channels, ecog_channels, metadata_str)

    with avg_psd_tab:
        render_average_psd_tab(
            trial_data, block_data, lfp_channels, ecog_channels, metadata_str
        )

    with multi_psd_tab:
        render_multi_participant_psd_analysis(lfp_channels + ecog_channels, "LFP+ECoG")


def render_multi_participant_psd_analysis(channels, channel_type):
    """
    Renders PSD comparison across all participants for selected channels.
    """
    if not channels:
        return

    st.markdown("### Multi-Participant PSD Comparison (All Participants)")

    from utils.data_loader import get_all_participants, load_participant_data
    from dashboard.backbone import PARTICIPANT_COLORS, PLOT_COLOR, LINE_STYLE
    from utils.plots import _create_base_figure

    selected_dataset = st.session_state.get("selected_dataset")
    participants = get_all_participants(selected_dataset)

    if not participants:
        st.warning("No participants found.")
        return

    target_channel = st.selectbox(
        "Select channel for comparison", channels, key="multi_participant_psd_chan"
    )

    psd_results = {}  # {p_id: {'on': mean_psd, 'off': mean_psd, 'freqs': freqs}}

    with st.spinner("Computing multi-participant PSDs..."):
        for p_id in participants:
            try:
                # Load only required columns for participant to aggregate
                cols_to_load = [
                    f"{target_channel}_psd_values",
                    f"{target_channel}_psd_freq",
                    "stim",
                ]
                p_df = load_participant_data(
                    p_id, selected_dataset, columns=cols_to_load
                )

                if p_df.is_empty():
                    continue

                if f"{target_channel}_psd_values" not in p_df.columns:
                    continue

                on_df = p_df.filter(pl.col("stim") == "on")
                off_df = p_df.filter(pl.col("stim") == "off")

                freqs = None
                if not p_df.is_empty():
                    freqs = np.array(p_df[f"{target_channel}_psd_freq"][0])

                on_mean = None
                off_mean = None

                if not on_df.is_empty():
                    vals = np.vstack(
                        [
                            np.array(x)
                            for x in on_df[f"{target_channel}_psd_values"].to_list()
                        ]
                    )
                    if vals.size > 0:
                        on_mean = 10 * np.log10(vals.mean(axis=0)) + 120

                if not off_df.is_empty():
                    vals = np.vstack(
                        [
                            np.array(x)
                            for x in off_df[f"{target_channel}_psd_values"].to_list()
                        ]
                    )
                    if vals.size > 0:
                        off_mean = 10 * np.log10(vals.mean(axis=0)) + 120

                psd_results[p_id] = {"on": on_mean, "off": off_mean, "freqs": freqs}

            except Exception as e:
                # Silently skip if error to avoid clutter
                continue

    # Plotting
    if not psd_results:
        st.info("No data available for multi-participant comparison.")
        return

    fig = _create_base_figure("", "Frequency (Hz)", "Power/Frequency (dB/Hz)")

    for p_id, res in psd_results.items():
        if p_id in PARTICIPANT_COLORS:
            colors = PARTICIPANT_COLORS[p_id]
            c_on = colors["dbs_on"] if isinstance(colors, dict) else colors.dbs_on
            c_off = colors["dbs_off"] if isinstance(colors, dict) else colors.dbs_off
        else:
            c_on = PLOT_COLOR.stim_on
            c_off = PLOT_COLOR.stim_off

        freqs = res["freqs"]

        if res["on"] is not None and freqs is not None:
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=res["on"],
                    mode="lines",
                    name=f"{p_id} ON",
                    line=dict(color=c_on, width=1.2),
                )
            )

        if res["off"] is not None and freqs is not None:
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=res["off"],
                    mode="lines",
                    name=f"{p_id} OFF",
                    line=dict(color=c_off, width=1.2, dash=LINE_STYLE.secondary),
                )
            )

    st.plotly_chart(fig, use_container_width=True)
