import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import json
import polars as pl
import plotly.express as px
import plotly.graph_objects as go

from dashboard.subtabs import list_variants, list_run_timestamps, config_for_variant
from utils.config import get_config
from utils.feature_importance import (
    load_model_and_compute_importance,
    normalize_importance,
)


def _find_model_path(variant_dir: Path, run_ts: str) -> Path:
    run_dir = variant_dir / run_ts
    model_files = list(run_dir.glob("model_*.pkl"))
    if model_files:
        return model_files[0]
    model_files = list(run_dir.glob("*.pkl"))
    return model_files[0] if model_files else None


def _load_run_metadata(variant_dir: Path, run_ts: str) -> dict | None:
    metadata_path = variant_dir / f"model_{run_ts}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return None


def _load_run_pearsonr(variant_dir: Path, run_ts: str) -> float | None:
    metadata_path = variant_dir / f"model_{run_ts}_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            if "r_mean_Z" in meta and meta["r_mean_Z"] is not None:
                return float(meta["r_mean_Z"])
        except Exception:
            pass

    results_path = variant_dir / f"val_results_{run_ts}"
    if results_path.exists():
        try:
            df = pl.read_parquet(results_path)
            for col in [
                "metric_pearson_r_mean_Z",
                "pearsonr_mean_Z",
                "pearson_overall_mean_Z",
            ]:
                if col in df.columns:
                    vals = df[col].to_list()
                    if vals and vals[0] is not None:
                        return float(vals[0])
        except Exception:
            pass
    return None


def _render_importance_bar_chart(
    importance_dict: dict, title: str = "Feature Importance"
):
    df = pd.DataFrame(
        [
            {"Feature": k, "Importance": v}
            for k, v in sorted(
                importance_dict.items(), key=lambda x: x[1], reverse=True
            )
        ]
    )

    fig = px.bar(
        df,
        y="Feature",
        x="Importance",
        orientation="h",
        title=title,
    )
    fig.update_layout(
        height=max(400, len(df) * 25),
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Importance Score (L2 norm of K projection)",
        yaxis_title="Neural Feature",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_rankings_table(importance_dict: dict):
    normalized = normalize_importance(importance_dict)
    df = pd.DataFrame(
        [
            {
                "Rank": i + 1,
                "Feature": k,
                "Importance": v,
                "Normalized": normalized[k],
            }
            for i, (k, v) in enumerate(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_performance_summary(results_path: Path, run_ts: str):
    import pickle

    val_results = results_path / f"val_results_{run_ts}"
    if not val_results.exists():
        val_pkl = results_path / run_ts / "val_results.pkl"
        if val_pkl.exists():
            with open(val_pkl, "rb") as f:
                res = pickle.load(f)
            if "pearson_overall_mean_Z" in res:
                corrs = res["pearson_overall_mean_Z"]
                if corrs is not None and len(corrs) > 0:
                    st.subheader("Behavioral Prediction Performance")
                    fig = go.Figure(data=[go.Bar(x=list(range(len(corrs))), y=corrs)])
                    fig.update_layout(
                        xaxis_title="Output Channel",
                        yaxis_title="Pearson Correlation",
                        title="Validation Set Correlations",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    return
    st.info("No validation results available for performance summary.")


def _render_grid_search_results(variant_dir: Path, runs: list):
    st.subheader("Grid Search Results")
    st.markdown(
        "Comparing Pearson R across different nx and n1 parameter combinations."
    )

    all_metadata = []
    for run_ts in runs:
        metadata = _load_run_metadata(variant_dir, run_ts)
        if metadata:
            metadata["run_ts"] = run_ts
            all_metadata.append(metadata)

    if not all_metadata:
        st.info(
            "No grid search results found. Run training with different nx/n1 values first."
        )
        return

    st.markdown(f"Found **{len(all_metadata)}** runs with metadata.")

    tab_overview, tab_neural, tab_behavioral = st.tabs(
        [
            "Overview (Mean)",
            "Per Neural Channel (Y)",
            "Per Behavioral Channel (Z)",
        ]
    )

    with tab_overview:
        _render_overview_plots(all_metadata)

    with tab_neural:
        _render_per_channel_plots(all_metadata, channel_type="Y")

    with tab_behavioral:
        _render_per_channel_plots(all_metadata, channel_type="Z")


def _render_overview_plots(all_metadata: list):
    """Render overview plots for mean Pearson R values."""
    grid_data = []
    for meta in all_metadata:
        r_mean_Y = meta.get("r_mean_Y")
        r_mean_Z = meta.get("r_mean_Z")

        if r_mean_Y is not None or r_mean_Z is not None:
            grid_data.append(
                {
                    "run_ts": meta["run_ts"],
                    "nx": meta.get("nx"),
                    "n1": meta.get("n1"),
                    "i": meta.get("i"),
                    "Pearson R (Y)": r_mean_Y,
                    "Pearson R (Z)": r_mean_Z,
                }
            )

    if not grid_data:
        st.info("No mean Pearson R data available.")
        return

    df = pd.DataFrame(grid_data)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Neural Prediction (Y)")
        if df["Pearson R (Y)"].notna().any():
            fig = px.scatter(
                df.dropna(subset=["Pearson R (Y)"]),
                x="nx",
                y="n1",
                color="Pearson R (Y)",
                hover_data=["run_ts", "nx", "n1", "i", "Pearson R (Y)"],
                title="n1 vs nx colored by Pearson R (Y)",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(
                xaxis_title="nx (State Dimension)",
                yaxis_title="n1 (Behavior Subspace)",
                height=450,
            )
            fig.update_traces(marker=dict(size=14))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No neural (Y) Pearson R data.")

    with col2:
        st.markdown("#### Behavioral Prediction (Z)")
        if df["Pearson R (Z)"].notna().any():
            fig = px.scatter(
                df.dropna(subset=["Pearson R (Z)"]),
                x="nx",
                y="n1",
                color="Pearson R (Z)",
                hover_data=["run_ts", "nx", "n1", "i", "Pearson R (Z)"],
                title="n1 vs nx colored by Pearson R (Z)",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(
                xaxis_title="nx (State Dimension)",
                yaxis_title="n1 (Behavior Subspace)",
                height=450,
            )
            fig.update_traces(marker=dict(size=14))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No behavioral (Z) Pearson R data.")

    with st.expander("View all results"):
        st.dataframe(
            df.sort_values("Pearson R (Z)", ascending=False, na_position="last"),
            use_container_width=True,
            hide_index=True,
        )


def _render_per_channel_plots(all_metadata: list, channel_type: str = "Z"):
    """Render per-channel Pearson R plots with stats."""
    key_prefix = "r_per_channel_" + channel_type
    channel_key = "output_channels" if channel_type == "Z" else "input_channels"
    type_label = "Behavioral" if channel_type == "Z" else "Neural"

    all_channels = set()
    for meta in all_metadata:
        if key_prefix in meta and meta[key_prefix]:
            all_channels.update(meta[key_prefix].keys())

    if not all_channels:
        st.info(
            f"No per-channel Pearson R data available for {type_label.lower()} channels. "
            "Run fast training to generate this data."
        )
        return

    st.markdown(f"**{len(all_channels)} {type_label} Channels Found**")

    selected_channel = st.selectbox(
        f"Select {type_label} Channel",
        options=sorted(all_channels),
        key=f"channel_select_{channel_type}",
    )

    channel_data = []
    for meta in all_metadata:
        r_per_channel = meta.get(key_prefix, {})
        if selected_channel in r_per_channel:
            ch_stats = r_per_channel[selected_channel]
            if isinstance(ch_stats, dict):
                r_mean = ch_stats.get("mean")
                r_std = ch_stats.get("std")
                r_min = ch_stats.get("min")
                r_max = ch_stats.get("max")
                n_trials = ch_stats.get("n_trials")
            else:
                r_mean = ch_stats
                r_std = r_min = r_max = n_trials = None

            if r_mean is not None:
                channel_data.append(
                    {
                        "run_ts": meta["run_ts"],
                        "nx": meta.get("nx"),
                        "n1": meta.get("n1"),
                        "i": meta.get("i"),
                        "Pearson R": r_mean,
                        "std": r_std,
                        "min": r_min,
                        "max": r_max,
                        "n_trials": n_trials,
                    }
                )

    if not channel_data:
        st.info(f"No data for channel '{selected_channel}'.")
        return

    df = pd.DataFrame(channel_data)

    fig = px.scatter(
        df,
        x="nx",
        y="n1",
        color="Pearson R",
        hover_data=[
            "run_ts",
            "nx",
            "n1",
            "i",
            "Pearson R",
            "std",
            "min",
            "max",
            "n_trials",
        ],
        title=f"n1 vs nx colored by Pearson R for '{selected_channel}'",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(
        xaxis_title="nx (State Dimension)",
        yaxis_title="n1 (Behavior Subspace)",
        height=500,
    )
    fig.update_traces(marker=dict(size=14))
    st.plotly_chart(fig, use_container_width=True)

    if df["std"].notna().any():
        st.markdown(f"**Stats for '{selected_channel}'**")
        st.dataframe(
            df[
                ["run_ts", "nx", "n1", "Pearson R", "std", "min", "max", "n_trials"]
            ].sort_values("Pearson R", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Compare all channels"):
        comparison_data = []
        for meta in all_metadata:
            r_per_channel = meta.get(key_prefix, {})
            for ch, ch_stats in r_per_channel.items():
                if isinstance(ch_stats, dict):
                    r_mean = ch_stats.get("mean")
                    r_std = ch_stats.get("std")
                    r_min = ch_stats.get("min")
                    r_max = ch_stats.get("max")
                else:
                    r_mean = ch_stats
                    r_std = r_min = r_max = None

                if r_mean is not None:
                    comparison_data.append(
                        {
                            "run_ts": meta["run_ts"],
                            "nx": meta.get("nx"),
                            "n1": meta.get("n1"),
                            "channel": ch,
                            "mean": r_mean,
                            "std": r_std if r_std else 0,
                            "min": r_min if r_min else r_mean,
                            "max": r_max if r_max else r_mean,
                        }
                    )

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            channels = sorted(comp_df["channel"].unique())
            colors = px.colors.qualitative.Plotly
            fig = go.Figure()

            for i, ch in enumerate(channels):
                ch_data = comp_df[comp_df["channel"] == ch]
                color = colors[i % len(colors)]

                fig.add_trace(
                    go.Box(
                        y=ch_data["mean"],
                        name=ch,
                        boxpoints="all",
                        jitter=0.3,
                        pointpos=0,
                        marker=dict(
                            color=color,
                            size=6,
                            opacity=0.7,
                        ),
                        line=dict(color=color, width=1.5),
                        fillcolor=(
                            color.replace("rgb", "rgba").replace(")", ", 0.3)")
                            if "rgb" in color
                            else color
                        ),
                        hoverinfo="y+name",
                    )
                )

            fig.update_layout(
                title=dict(
                    text=f"Pearson R Distribution by {type_label} Channel",
                    font=dict(size=14),
                ),
                xaxis_title=f"{type_label} Channel",
                yaxis_title="Pearson R",
                height=450,
                showlegend=False,
                template="plotly_white",
                font=dict(family="Inter, sans-serif"),
                margin=dict(l=60, r=40, t=60, b=100),
            )
            fig.update_yaxes(
                zeroline=True, zerolinecolor="rgba(0,0,0,0.2)", zerolinewidth=1
            )


            st.plotly_chart(fig, use_container_width=True)

            summary_df = (
                comp_df.groupby("channel")
                .agg(
                    mean_r=("mean", "mean"),
                    std_r=("mean", "std"),
                    min_r=("mean", "min"),
                    max_r=("mean", "max"),
                    n_runs=("mean", "count"),
                )
                .reset_index()
            )
            summary_df = summary_df.sort_values("mean_r", ascending=False)
            summary_df.columns = ["Channel", "Mean R", "Std", "Min", "Max", "N Runs"]

            st.markdown("**Summary Statistics Across Runs**")
            st.dataframe(
                summary_df.style.format(
                    {
                        "Mean R": "{:.4f}",
                        "Std": "{:.4f}",
                        "Min": "{:.4f}",
                        "Max": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


def feature_importance_tab(project_root: Path):
    st.header("Feature Importance Analysis")

    RESULTS_ROOT = project_root / "results"

    variants = list_variants(RESULTS_ROOT)
    if not variants:
        st.info("No model variants found. Train a model first.")
        return

    variant = st.selectbox("Model variant", options=variants, key="fi_variant")
    variant_dir = RESULTS_ROOT / variant
    runs = list_run_timestamps(variant_dir)

    if not runs:
        st.info("No runs found for this variant.")
        return

    run_ts = st.selectbox("Run timestamp", options=runs, key="fi_run")
    cfg_path = config_for_variant(project_root, variant)

    if cfg_path is None:
        st.error(f"Config not found for variant '{variant}'.")
        return

    cfg = get_config(str(cfg_path))
    input_channels = cfg.data.channels.neural_input

    if not input_channels:
        st.error("No input channels defined in config.")
        return

    model_path = _find_model_path(variant_dir, run_ts)

    if model_path is None:
        st.error("No model file found.")
        return

    if st.button("Compute Feature Importance", key="fi_compute"):
        st.session_state["fi_computed"] = (str(model_path), input_channels)

    if st.session_state.get("fi_computed"):
        cached_path, cached_channels = st.session_state["fi_computed"]
        if cached_path == str(model_path):
            with st.spinner("Computing feature importance..."):
                try:
                    importance, ranked = load_model_and_compute_importance(
                        model_path, cached_channels
                    )

                    st.success(f"Analyzed {len(importance)} neural features.")

                    tab_chart, tab_table, tab_perf = st.tabs(
                        [
                            "Importance Chart",
                            "Rankings Table",
                            "Performance Summary",
                        ]
                    )

                    with tab_chart:
                        _render_importance_bar_chart(importance)

                    with tab_table:
                        _render_rankings_table(importance)

                    with tab_perf:
                        _render_performance_summary(variant_dir, run_ts)

                except Exception as e:
                    st.error(f"Error computing importance: {e}")

