import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Any


def _load_summaries(results_root: Path) -> Dict[str, List[Dict]]:
    """Load all data_hungriness_summary.json files from results/data_hungriness/."""
    dh_root = results_root / "data_hungriness"
    summaries = {}
    if not dh_root.exists():
        return summaries
    for model_dir in sorted(dh_root.iterdir()):
        if not model_dir.is_dir():
            continue
        summary_path = model_dir / "data_hungriness_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            summaries[model_dir.name] = data
    return summaries


def _extract_participant_session(model_name: str) -> str:
    """Extract participant-session identifier from model name like
    'psid_behavioral_PDI1_2_nx_40_...' -> 'PDI1_S2'."""
    parts = model_name.split("_")
    for i, p in enumerate(parts):
        if p.startswith("PDI") and i + 1 < len(parts):
            return f"{p}_S{parts[i+1]}"
    return model_name


def _create_feature_plot(
    summaries_for_ps: Dict[str, List[Dict]],
    feature_type: str,
    title: str,
) -> go.Figure:
    """Create a line plot for either neural or behavioral features.
    Each line = one feature channel, averaged across models if multiple for same participant-session.
    """
    key_per_channel = f"fisher_z_{feature_type}_per_channel"

    # Collect all unique channels and all data points
    all_channels = set()
    for model_name, data in summaries_for_ps.items():
        for step in data:
            if step.get(key_per_channel):
                all_channels.update(step[key_per_channel].keys())

    if not all_channels:
        return None

    fig = go.Figure()

    # Use a color cycle
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
        "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    ]

    for ch_idx, channel in enumerate(sorted(all_channels)):
        # Average across models for this participant-session
        trial_counts = []
        z_values = []

        # Gather all data points grouped by train_pct
        pct_to_vals = {}
        pct_to_trials = {}
        for model_name, data in summaries_for_ps.items():
            for step in data:
                pct = step["train_pct"]
                ch_data = step.get(key_per_channel, {})
                if channel in ch_data and ch_data[channel] is not None:
                    if pct not in pct_to_vals:
                        pct_to_vals[pct] = []
                        pct_to_trials[pct] = step["n_train_trials"]
                    pct_to_vals[pct].append(ch_data[channel])

        if not pct_to_vals:
            continue

        # Sort by train_pct
        sorted_pcts = sorted(pct_to_vals.keys())
        x_vals = [pct_to_trials.get(p, 0) for p in sorted_pcts]
        y_vals = [float(np.mean(pct_to_vals[p])) for p in sorted_pcts]

        color = colors[ch_idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=channel,
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{channel}</b><br>"
                "Trials: %{x}<br>"
                "Fisher Z: %{customdata:.4f}<extra></extra>"
            ),
            customdata=y_vals,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Number of Training Trials",
        yaxis_title="Fisher Z-scored Pearson R",
        template="plotly_dark",
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
        ),
        margin=dict(r=250),
        hovermode="x unified",
    )

    return fig


def render_data_hungriness_tab(project_root, results_root=None):
    """Render the Data Hungriness analysis tab."""
    st.subheader("Data Hungriness Analysis")
    RESULTS_ROOT = results_root if results_root else project_root / "results"
    summaries = _load_summaries(RESULTS_ROOT)

    if not summaries:
        st.info(
            "No data hungriness results found. "
            "Run `bash scripts/run_all_data_hungriness.sh` to generate results."
        )
        return

    # Group by participant-session
    ps_groups: Dict[str, Dict[str, List[Dict]]] = {}
    for model_name, data in summaries.items():
        ps = _extract_participant_session(model_name)
        if ps not in ps_groups:
            ps_groups[ps] = {}
        ps_groups[ps][model_name] = data

    for ps_name in sorted(ps_groups.keys()):
        st.markdown(f"### {ps_name}")
        models_data = ps_groups[ps_name]

        col1, col2 = st.columns(2)

        with col1:
            neural_fig = _create_feature_plot(
                models_data, "neural", f"Neural Features — {ps_name}"
            )
            if neural_fig:
                st.plotly_chart(neural_fig, use_container_width=True)
            else:
                st.info("No neural feature data available.")

        with col2:
            behavioral_fig = _create_feature_plot(
                models_data, "behavioral", f"Behavioral Features — {ps_name}"
            )
            if behavioral_fig:
                st.plotly_chart(behavioral_fig, use_container_width=True)
            else:
                st.info("No behavioral feature data available.")
