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


def _load_run_r2(variant_dir: Path, run_ts: str) -> float | None:
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
            for col in ["metric_pearson_r_mean_Z", "pearsonr_mean_Z", "pearson_overall_mean_Z"]:
                if col in df.columns:
                    vals = df[col].to_list()
                    if vals and vals[0] is not None:
                        return float(vals[0])
        except Exception:
            pass
    return None


def _render_importance_bar_chart(importance_dict: dict, title: str = "Feature Importance"):
    df = pd.DataFrame([
        {"Feature": k, "Importance": v}
        for k, v in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    ])

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
    df = pd.DataFrame([
        {
            "Rank": i + 1,
            "Feature": k,
            "Importance": v,
            "Normalized": normalized[k],
        }
        for i, (k, v) in enumerate(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
    ])
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
                    fig = go.Figure(data=[
                        go.Bar(x=list(range(len(corrs))), y=corrs)
                    ])
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
    st.markdown("Comparing R2 (Zp) across different nx and n1 parameter combinations.")
    
    grid_data = []
    for run_ts in runs:
        metadata = _load_run_metadata(variant_dir, run_ts)
        r2 = _load_run_r2(variant_dir, run_ts)
        
        if metadata and r2 is not None:
            grid_data.append({
                "run_ts": run_ts,
                "nx": metadata.get("nx"),
                "n1": metadata.get("n1"),
                "i": metadata.get("i"),
                "R2": r2,
            })
    
    if not grid_data:
        st.info("No grid search results found. Run training with different nx/n1 values first.")
        return
    
    df = pd.DataFrame(grid_data)
    st.markdown(f"Found **{len(df)}** runs with metadata.")
    
    fig = px.scatter(
        df,
        x="nx",
        y="R2",
        color="n1",
        hover_data=["run_ts", "nx", "n1", "i", "R2"],
        title="R2 vs nx (colored by n1)",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        xaxis_title="nx (State Dimension)",
        yaxis_title="R2 (Behavioral Prediction)",
        height=500,
    )
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View all results"):
        st.dataframe(
            df.sort_values("R2", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def feature_importance_tab(project_root: Path):
    st.header("Feature Importance Analysis")
    st.markdown(
        "Analyze which neural features (channels × bands) are most predictive of behavior "
        "based on the PSID Kalman gain matrix K."
    )

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

    main_tab_single, main_tab_grid = st.tabs([
        "Single Run Analysis",
        "Grid Search Results",
    ])
    
    with main_tab_grid:
        _render_grid_search_results(variant_dir, runs)
    
    with main_tab_single:
        run_ts = st.selectbox("Run timestamp", options=runs, key="fi_run")
        cfg_path = config_for_variant(project_root, variant)

        if cfg_path is None:
            st.error(f"Config not found for variant '{variant}'.")
            return

        cfg = get_config(str(cfg_path))
        input_channels = cfg.data.channels.input

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

                        tab_chart, tab_table, tab_perf = st.tabs([
                            "Importance Chart",
                            "Rankings Table",
                            "Performance Summary",
                        ])

                        with tab_chart:
                            _render_importance_bar_chart(importance)

                        with tab_table:
                            _render_rankings_table(importance)

                        with tab_perf:
                            _render_performance_summary(variant_dir, run_ts)

                    except Exception as e:
                        st.error(f"Error computing importance: {e}")

