import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from pathlib import Path


def grid_search_tab(project_root: Path):
    """Render the grid search results tab."""
    st.header("Grid Search Results Explorer")
    st.markdown(
        "Explore PSID grid search results. Select a participant and session to analyze hyper-parameters."
    )

    results_dir = project_root / "results" / "psid_grid_search"
    parquet_path = results_dir / "results.parquet"

    if not parquet_path.exists():
        st.info(
            "No grid search results found in `results/psid_grid_search/results.parquet`. "
            "Run a grid search first:\n\n"
            "```bash\n"
            "python -m training.psid_grid_search --config training/setups/psid_grid_search.yaml\n"
            "```"
        )
        return

    try:
        q = pl.scan_parquet(parquet_path)
        
        with st.spinner("Fetching available participants and sessions..."):
            partition_info = q.select(["participant_id", "session"]).unique().collect()
        
        participants = sorted(partition_info["participant_id"].unique().to_list())
        
        col_p, col_s = st.columns(2)
        
        with col_p:
            selected_participant = st.selectbox("Select Participant", participants)
            
        sessions = sorted(
            partition_info.filter(pl.col("participant_id") == selected_participant)["session"]
            .unique()
            .to_list()
        )
        
        with col_s:
            selected_session = st.selectbox("Select Session", sessions)

        if f"gs_df" not in st.session_state or st.session_state.get("gs_selection") != (selected_participant, selected_session):
            if st.button("Load Results", type="primary"):
                with st.spinner(f"Loading results for P{selected_participant} S{selected_session}..."):
                    df_pl = q.filter(
                        (pl.col("participant_id") == selected_participant) & 
                        (pl.col("session") == str(selected_session))
                    ).collect()
                    
                    df = df_pl.to_pandas()
                    st.session_state["gs_df"] = df
                    st.session_state["gs_selection"] = (selected_participant, selected_session)
                    st.rerun()
            else:
                st.info("Click 'Load Results' to fetch and analyze grid search data for the selection.")
                # If we have old data, show a message that it's from a different selection or just stop
                return
        
        df = st.session_state["gs_df"]
        
        if len(df) == 0:
            st.warning(f"No results found for Participant {selected_participant}, Session {selected_session}")
            # Clear invalid state
            del st.session_state["gs_df"]
            return

        st.success(f"Loaded {len(df)} runs for P{selected_participant} S{selected_session}")

    except Exception as e:
        st.error(f"Error loading results: {e}")
        return

    # Identify column types
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

    # Filters
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

        if "neural_bands" in df.columns:
            # Convert to string to handle unhashable types like lists/arrays
            nb_str_col = df["neural_bands"].apply(lambda x: str(list(x)) if hasattr(x, "__iter__") and not isinstance(x, str) else str(x))
            nb_vals = sorted(nb_str_col.dropna().unique())
            selected_nb = st.multiselect("neural_bands", nb_vals, default=nb_vals)
            filtered_df = filtered_df[nb_str_col.isin(selected_nb)]

    st.markdown(f"**{len(filtered_df)} configurations** after filtering")

    # Results table
    st.subheader("Comprehensive Results")
    
    # Column ordering: params first, then metrics
    display_cols = param_cols + metric_cols
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    # Format numeric columns
    styled_df = filtered_df[display_cols].copy()
    for col in metric_cols:
        if col in styled_df.columns:
            styled_df[col] = styled_df[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )

    st.dataframe(styled_df, use_container_width=True, height=400)

    # Distribution plots
    st.subheader("Metric Distributions")

    available_metrics = [c for c in metric_cols if c in filtered_df.columns and filtered_df[c].notna().any()]

    if available_metrics:
        plot_cols = st.columns(2)

        with plot_cols[0]:
            selected_metric = st.selectbox("Select Metric", available_metrics, index=0)

        with plot_cols[1]:
            plot_type = st.radio("Plot Type", ["Histogram", "Box Plot"], horizontal=True)

        if plot_type == "Histogram":
            fig = px.histogram(
                filtered_df,
                x=selected_metric,
                nbins=20,
                title=f"Distribution of {selected_metric}",
                color_discrete_sequence=["#7c3aed"]
            )
        else:
            fig = px.box(
                filtered_df,
                y=selected_metric,
                title=f"Distribution of {selected_metric}",
                color_discrete_sequence=["#00d4ff"]
            )

        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Parameter vs metric scatter
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
                ["None"] + [c for c in param_cols if c in filtered_df.columns and c != x_param],
                index=0,
            )

            fig = px.scatter(
                filtered_df,
                x=x_param,
                y=y_metric,
                color=color_param if color_param != "None" else None,
                hover_data=filtered_df.columns.tolist(),
                title=f"{y_metric} vs {x_param}",
                size_max=15
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # Export
    st.subheader("Export Clean Data")
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        "Download Filtered Results (CSV)",
        csv_data,
        file_name=f"psid_grid_search_{selected_participant}_S{selected_session}.csv",
        mime="text/csv",
    )

