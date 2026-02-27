import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from pathlib import Path
from dashboard.backbone import render_styled_table


def _find_parquet_source(results_dir: Path) -> Path | None:
    """
    Find a readable parquet source inside a grid search group directory.

    Supports:
      1. results.parquet  (single file – preferred)
      2. results_parquet/ (partitioned directory with *.parquet leaves)
    """
    rp = results_dir / "results.parquet"
    if rp.is_file():
        return rp

    parquet_dir = results_dir / "results_parquet"
    if parquet_dir.is_dir() and any(parquet_dir.rglob("*.parquet")):
        return parquet_dir

    return None


def grid_search_tab(project_root: Path, results_root=None):
    """Render the grid search results tab."""
    st.header("Grid Search Results Explorer")
    st.markdown(
        "Explore PSID grid search results. Select a participant and session to analyze hyper-parameters."
    )

    RESULTS_ROOT = results_root if results_root else project_root / "results"

    if not RESULTS_ROOT.exists() or not RESULTS_ROOT.is_dir():
        st.info(f"Results directory `{RESULTS_ROOT}` not found.")
        return

    # Find all grid search result groups that have some parquet data
    all_groups = sorted(
        [
            d.name
            for d in RESULTS_ROOT.iterdir()
            if d.is_dir() and _find_parquet_source(d) is not None
        ]
    )

    if not all_groups:
        st.info("No grid search result groups found in `results/`.")
        return

    selected_group = st.selectbox(
        "Select Result Group",
        all_groups,
        index=(
            all_groups.index("psid_grid_search_part1")
            if "psid_grid_search_part1" in all_groups
            else 0
        ),
    )

    results_dir = RESULTS_ROOT / selected_group
    parquet_src = _find_parquet_source(results_dir)

    if parquet_src is None:
        st.info(
            "No grid search results found. Run a grid search first:\n\n"
            "```bash\n"
            "python -m training.psid_grid_search --config (path to your config under training/setups/)\n"
            "```"
        )
        return

    # ------------------------------------------------------------------
    # Load the full parquet eagerly, then filter in the UI
    # ------------------------------------------------------------------
    cache_key = f"gs_full_{selected_group}"
    if cache_key not in st.session_state:
        try:
            with st.spinner("Loading grid search results..."):
                if parquet_src.is_dir():
                    df_pl = pl.read_parquet(
                        parquet_src / "**/*.parquet",
                        hive_partitioning=True,
                    )
                else:
                    df_pl = pl.read_parquet(parquet_src)
                st.session_state[cache_key] = df_pl.to_pandas()
        except Exception as e:
            st.error(f"Error loading results: {e}")
            return

    df_full = st.session_state[cache_key]

    if len(df_full) == 0:
        st.warning("Parquet is empty — no results to display.")
        return

    # ------------------------------------------------------------------
    # Participant / session filter (if columns exist)
    # ------------------------------------------------------------------
    df = df_full.copy()

    has_participant = "participant_id" in df.columns
    has_session = "session" in df.columns

    if has_participant or has_session:
        filter_cols = st.columns(2)
        if has_participant:
            with filter_cols[0]:
                participants = sorted(df["participant_id"].dropna().unique())
                selected_participant = st.selectbox(
                    "Filter by Participant", ["All"] + participants
                )
                if selected_participant != "All":
                    df = df[df["participant_id"] == selected_participant]

        if has_session:
            with filter_cols[1]:
                sessions = sorted(df["session"].dropna().unique().astype(str))
                selected_session = st.selectbox("Filter by Session", ["All"] + sessions)
                if selected_session != "All":
                    df = df[df["session"].astype(str) == selected_session]

    st.success(f"Showing **{len(df)}** of {len(df_full)} total runs")

    # ------------------------------------------------------------------
    # Identify column types
    # ------------------------------------------------------------------
    metric_cols = [
        c
        for c in df.columns
        if c
        in [
            "pearson_mean",
            "pearson_median",
            "pearson_trimmed",
            "pearson_fisher",
            "r_squared",
            "xcorr_mean",
            "xcorr_median",
            "xcorr_lag_mean_ms",
            "xcorr_lag_median_ms",
            "cv",
            "pct_above_zero",
            "pct_above_03",
            "n_trials",
        ]
    ]
    param_cols = [
        c
        for c in df.columns
        if c
        in [
            "nx",
            "n1",
            "alpha_Q",
            "alpha_R",
            "backward_kalman",
            "rescale_states",
            "max_eigenvalue",
            "neural_bands",
            "behavioral_outputs",
        ]
    ]
    config_cols = [
        c
        for c in df.columns
        if c
        in [
            "neural_input",
            "behavioral_output",
        ]
    ]

    # ------------------------------------------------------------------
    # Filters on parameters
    # ------------------------------------------------------------------
    st.subheader("Fine-tune Filters")
    col1, col2, col3 = st.columns(3)

    filtered_df = df.copy()

    with col1:
        if "nx" in df.columns:
            nx_vals = sorted(df["nx"].dropna().unique())
            selected_nx = st.multiselect("nx", nx_vals, default=nx_vals)
            filtered_df = filtered_df[filtered_df["nx"].isin(selected_nx)]

        if "n1" in df.columns:
            n1_vals = sorted(df["n1"].dropna().unique())
            selected_n1 = st.multiselect("n1", n1_vals, default=n1_vals)
            filtered_df = filtered_df[filtered_df["n1"].isin(selected_n1)]

    with col2:
        if "alpha_Q" in df.columns:
            aq_vals = sorted(df["alpha_Q"].dropna().unique())
            selected_aq = st.multiselect("alpha_Q", aq_vals, default=aq_vals)
            filtered_df = filtered_df[filtered_df["alpha_Q"].isin(selected_aq)]

        if "alpha_R" in df.columns:
            ar_vals = sorted(df["alpha_R"].dropna().unique())
            selected_ar = st.multiselect("alpha_R", ar_vals, default=ar_vals)
            filtered_df = filtered_df[filtered_df["alpha_R"].isin(selected_ar)]

    with col3:
        if "backward_kalman" in df.columns:
            bk_vals = list(df["backward_kalman"].dropna().unique())
            selected_bk = st.multiselect("backward_kalman", bk_vals, default=bk_vals)
            filtered_df = filtered_df[filtered_df["backward_kalman"].isin(selected_bk)]

    st.markdown(f"**{len(filtered_df)} configurations** after filtering")

    # ------------------------------------------------------------------
    # Results table (params + config + metrics)
    # ------------------------------------------------------------------
    st.subheader("Comprehensive Results")

    display_cols = param_cols + config_cols + metric_cols
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    # Format numeric columns
    styled_df = filtered_df[display_cols].copy()
    for col in metric_cols:
        if col in styled_df.columns:
            styled_df[col] = styled_df[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )

    render_styled_table(styled_df, key="tbl_grid_search_results")

    # ------------------------------------------------------------------
    # Distribution plots
    # ------------------------------------------------------------------
    st.subheader("Metric Distributions")

    available_metrics = [
        c
        for c in metric_cols
        if c in filtered_df.columns and filtered_df[c].notna().any()
    ]

    if available_metrics:
        plot_cols = st.columns(2)

        with plot_cols[0]:
            selected_metric = st.selectbox("Select Metric", available_metrics, index=0)

        with plot_cols[1]:
            plot_type = st.radio(
                "Plot Type", ["Histogram", "Box Plot"], horizontal=True
            )

        if plot_type == "Histogram":
            fig = px.histogram(
                filtered_df,
                x=selected_metric,
                nbins=20,
                title=f"Distribution of {selected_metric}",
                color_discrete_sequence=["#7c3aed"],
            )
        else:
            fig = px.box(
                filtered_df,
                y=selected_metric,
                title=f"Distribution of {selected_metric}",
                color_discrete_sequence=["#00d4ff"],
            )

        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------------
        # Heatmap Explorer
        # ------------------------------------------------------------------
        st.subheader("Heatmap Explorer")

        heatmap_axis_candidates = [
            c for c in param_cols if c in filtered_df.columns
        ] + [c for c in config_cols if c in filtered_df.columns]

        if len(heatmap_axis_candidates) >= 2 and selected_metric:
            hm_cols = st.columns(2)
            with hm_cols[0]:
                hm_x = st.selectbox(
                    "Heatmap X-axis", heatmap_axis_candidates, index=0, key="hm_x"
                )
            with hm_cols[1]:
                remaining = [c for c in heatmap_axis_candidates if c != hm_x]
                hm_y = st.selectbox("Heatmap Y-axis", remaining, index=0, key="hm_y")

            try:
                pivot = filtered_df.pivot_table(
                    index=hm_y, columns=hm_x, values=selected_metric, aggfunc="mean"
                )
                fig_heat = go.Figure(
                    data=go.Heatmap(
                        z=pivot.values,
                        x=[str(c) for c in pivot.columns],
                        y=[str(r) for r in pivot.index],
                        colorscale="Viridis",
                        text=np.round(pivot.values, 4),
                        texttemplate="%{text}",
                        colorbar=dict(title=selected_metric),
                    )
                )
                fig_heat.update_layout(
                    title=f"{selected_metric}  ({hm_x} × {hm_y})",
                    xaxis_title=hm_x,
                    yaxis_title=hm_y,
                    height=500,
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not build heatmap: {e}")

        # ------------------------------------------------------------------
        # Parameter vs metric scatter
        # ------------------------------------------------------------------
        st.subheader("Parameter Explorer")

        scatter_cols = st.columns(2)
        with scatter_cols[0]:
            x_param = st.selectbox(
                "X-axis (parameter)",
                [c for c in param_cols if c in filtered_df.columns],
                index=0 if param_cols else None,
            )
        with scatter_cols[1]:
            y_metric = st.selectbox(
                "Y-axis (metric)",
                available_metrics,
                index=0,
                key="scatter_metric",
            )

        if x_param and y_metric:
            color_param = st.selectbox(
                "Color by",
                ["None"]
                + [c for c in param_cols if c in filtered_df.columns and c != x_param],
                index=0,
            )

            fig = px.scatter(
                filtered_df,
                x=x_param,
                y=y_metric,
                color=color_param if color_param != "None" else None,
                hover_data=filtered_df.columns.tolist(),
                title=f"{y_metric} vs {x_param}",
                size_max=15,
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    st.subheader("Export Clean Data")
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        "Download Filtered Results (CSV)",
        csv_data,
        file_name=f"psid_grid_search_{selected_group}.csv",
        mime="text/csv",
    )
