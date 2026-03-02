import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Any


def _load_summaries(results_root: Path) -> Dict[str, List[Dict]]:
    """Load all data_hungriness_summary.json files from results/data_hungriness/.
    Also checks for nested results/results/data_hungriness/ as a fallback.
    """
    search_paths = [
        results_root / "data_hungriness",
        results_root / "results" / "data_hungriness",
    ]
    summaries = {}

    for dh_root in search_paths:
        if not dh_root.exists():
            continue
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
    parts = model_name.split("_")
    for i, p in enumerate(parts):
        if p.startswith("PDI") and i + 1 < len(parts):
            return f"{p}_S{parts[i+1]}"
    return model_name


def _create_feature_plot(
    summaries_for_ps: Dict[str, List[Dict]],
    feature_type: str,
    metric_type: str,
    title: str,
) -> go.Figure:
    """Create a line plot for either neural or behavioral features.
    Each line = one feature channel, averaged across models if multiple for same participant-session.
    metric_type should be 'fisher_z' or 'rmse'.
    """
    mean_key = f"{metric_type}_{feature_type}_per_channel_mean"
    se_key = f"{metric_type}_{feature_type}_per_channel_se"

    # Collect all unique channels and all data points
    all_channels = set()
    for model_name, data in summaries_for_ps.items():
        for step in data:
            if step.get(mean_key):
                all_channels.update(step[mean_key].keys())

    if not all_channels:
        return None

    fig = go.Figure()

    # Use a color cycle
    colors = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#7F7F7F",
        "#BCBD22",
        "#17BECF",
    ]

    for ch_idx, channel in enumerate(sorted(all_channels)):
        # Average across models for this participant-session
        trial_counts = []
        z_values = []

        # Gather all data points grouped by train_pct
        pct_to_vals = {}
        pct_to_ses = {}
        pct_to_time = {}
        for model_name, data in summaries_for_ps.items():
            for step in data:
                pct = step["train_pct"]
                ch_mean_data = step.get(mean_key, {})
                ch_se_data = step.get(se_key, {})
                if channel in ch_mean_data and ch_mean_data[channel] is not None:
                    if pct not in pct_to_vals:
                        pct_to_vals[pct] = []
                        pct_to_ses[pct] = []
                        pct_to_time[pct] = step["n_train_trials"] * 9.0  # 9s per trial
                    pct_to_vals[pct].append(ch_mean_data[channel])
                    if channel in ch_se_data and ch_se_data[channel] is not None:
                        pct_to_ses[pct].append(ch_se_data[channel])
                    else:
                        pct_to_ses[pct].append(0.0)

        if not pct_to_vals:
            continue

        # Sort by train_pct
        sorted_pcts = sorted(pct_to_vals.keys())
        x_vals = [pct_to_time.get(p, 0.0) for p in sorted_pcts]
        y_vals = [float(np.mean(pct_to_vals[p])) for p in sorted_pcts]
        # For pooling SEs from different models, we'll just take the mean of SEs as an approximation
        se_vals = [float(np.mean(pct_to_ses[p])) for p in sorted_pcts]

        y_upper = [y + se for y, se in zip(y_vals, se_vals)]
        y_lower = [y - se for y, se in zip(y_vals, se_vals)]

        color = colors[ch_idx % len(colors)]

        # Add main line
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=channel,
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{channel}</b><br>"
                    "Time: %{x}s<br>"
                    "Value: %{customdata[0]:.4f} ± %{customdata[1]:.4f}<extra></extra>"
                ),
                customdata=list(zip(y_vals, se_vals)),
                legendgroup=channel,
            )
        )

        # Add error envelope (transparent)
        # Convert hex color to rgba for transparency
        if color.startswith("#"):
            r, g, b = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
            fillcolor = f"rgba({r},{g},{b},0.2)"
        else:
            fillcolor = "rgba(100,100,100,0.2)"

        fig.add_trace(
            go.Scatter(
                x=x_vals + x_vals[::-1],
                y=y_upper + y_lower[::-1],
                fill="toself",
                fillcolor=fillcolor,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=channel,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Training Time (seconds)",
        yaxis_title="Fisher Z Pearson R" if metric_type == "fisher_z" else "RMSE",
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

        # Pearson R Row
        st.markdown("#### Pearson R (Fisher Z)")
        col1, col2 = st.columns(2)
        with col1:
            neural_r_fig = _create_feature_plot(
                models_data, "neural", "fisher_z", f"Neural Features — {ps_name}"
            )
            if neural_r_fig:
                st.plotly_chart(neural_r_fig, use_container_width=True)
            else:
                st.info("No neural Pearson R data available.")
        with col2:
            behavioral_r_fig = _create_feature_plot(
                models_data,
                "behavioral",
                "fisher_z",
                f"Behavioral Features — {ps_name}",
            )
            if behavioral_r_fig:
                st.plotly_chart(behavioral_r_fig, use_container_width=True)
            else:
                st.info("No behavioral Pearson R data available.")

        # RMSE Row
        st.markdown("#### RMSE")
        col3, col4 = st.columns(2)
        with col3:
            neural_rmse_fig = _create_feature_plot(
                models_data, "neural", "rmse", f"Neural Features — {ps_name}"
            )
            if neural_rmse_fig:
                st.plotly_chart(neural_rmse_fig, use_container_width=True)
            else:
                st.info("No neural RMSE data available.")
        with col4:
            behavioral_rmse_fig = _create_feature_plot(
                models_data, "behavioral", "rmse", f"Behavioral Features — {ps_name}"
            )
            if behavioral_rmse_fig:
                st.plotly_chart(behavioral_rmse_fig, use_container_width=True)
            else:
                st.info("No behavioral RMSE data available.")

        st.markdown("---")
