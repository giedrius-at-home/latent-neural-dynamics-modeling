import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any
import re
from scipy.stats import ks_2samp, gaussian_kde, zscore, probplot
from scipy.signal import welch
from statsmodels.tsa.stattools import acf
from dashboard.subtabs.helpers import (
    list_variants,
    list_run_timestamps,
    load_precomputed_results,
    variant_short_name,
    find_baseline_variants,
    get_project_root,
)
from dashboard.backbone import (
    PALETTE,
    PLOT_STYLE,
    create_base_comparison_figure,
)

BASELINE_COLOR = "#00E5FF"
import pathlib
import re


def _get_groups_map(channel_names: list, use_bands: bool) -> list[str]:
    """Map each channel name to its display group (band name or raw channel name)."""
    if not use_bands:
        return list(channel_names)
    groups = []
    for name in channel_names:
        parts = str(name).split("_")
        if len(parts) >= 3:
            groups.append(parts[2].capitalize())
        elif len(parts) >= 2:
            groups.append(parts[1].capitalize())
        else:
            groups.append("Unknown")
    return groups


def _match_channels_by_name(
    main_channels: list, baseline_channels: list
) -> tuple[list[int], list[int]]:
    main_indices = []
    baseline_indices = []

    baseline_name_to_idx = {
        str(name): idx for idx, name in enumerate(baseline_channels)
    }

    for main_idx, main_name in enumerate(main_channels):
        main_name_str = str(main_name)
        if main_name_str in baseline_name_to_idx:
            main_indices.append(main_idx)
            baseline_indices.append(baseline_name_to_idx[main_name_str])

    return main_indices, baseline_indices


def render_cross_trial_performance_tab(
    split_res: Dict[str, Any], sampling_freq: float = 80.0, cfg_path: Any = None
):
    st.subheader("Performance Overview Across All Trials")

    pearson_y = split_res.get("pearson_per_channel", [])
    pearson_z = split_res.get("pearson_per_channel_Z", [])

    y_chan_names = split_res.get("input_channels", [])
    z_chan_names = split_res.get("output_channels", [])

    if not pearson_y and not pearson_z:
        st.info("No Pearson correlation results available for this split.")
        return

    neural_available = False
    if pearson_y and len(pearson_y) > 0:
        try:
            r_y_arr = np.array(pearson_y)
            if r_y_arr.ndim == 2:
                n_trials, n_chans = r_y_arr.shape
                if not y_chan_names or len(y_chan_names) != n_chans:
                    y_chan_names = [f"Ch{i}" for i in range(n_chans)]

                df_y = pd.DataFrame(r_y_arr, columns=y_chan_names)
                df_y["Trial"] = np.arange(n_trials)
                df_y["Mean_R"] = df_y[y_chan_names].mean(axis=1)
                neural_available = True
        except Exception as e:
            st.error(f"Error processing neural Pearson data: {e}")

    behavioral_available = False
    if pearson_z and len(pearson_z) > 0:
        try:
            r_z_arr = np.array(pearson_z)
            if r_z_arr.ndim == 2:
                n_trials, n_chans_z = r_z_arr.shape
                if not z_chan_names or len(z_chan_names) != n_chans_z:
                    z_chan_names = [f"Z_Ch{i}" for i in range(n_chans_z)]

                df_z = pd.DataFrame(r_z_arr, columns=z_chan_names)
                df_z["Trial"] = np.arange(n_trials)
                df_z["Mean_R"] = df_z[z_chan_names].mean(axis=1)
                behavioral_available = True
        except Exception as e:
            st.error(f"Error processing behavioral Pearson data: {e}")

    if not neural_available and not behavioral_available:
        st.info("Could not format Pearson data for heatmaps.")
        return

    baseline_r_y = None
    baseline_r_z = None
    baseline_yp = None
    baseline_zp = None
    baseline_name = "Baseline"
    model_name = "Model"

    if cfg_path:

        if isinstance(cfg_path, str):
            cfg_path = pathlib.Path(cfg_path)
        project_root = get_project_root(cfg_path)
        results_root = project_root / "results"
        all_variants = list_variants(results_root)

        current_variant = cfg_path.stem
        model_name = variant_short_name(current_variant)

        baseline_variants = find_baseline_variants(current_variant, all_variants)

        if baseline_variants:
            selected_baseline = st.selectbox(
                "Select Baseline Model",
                options=["None"] + baseline_variants,
                index=1 if baseline_variants else 0,
                key="global_baseline_select",
            )

            if selected_baseline != "None":
                baseline_dir = results_root / selected_baseline
                baseline_timestamps = list_run_timestamps(baseline_dir)

                if baseline_timestamps:
                    baseline_ts = st.selectbox(
                        "Baseline run timestamp",
                        options=baseline_timestamps,
                        index=len(baseline_timestamps) - 1,
                        key="global_baseline_ts_select",
                    )
                    split_name = st.session_state.get("pred_split", "val")
                    baseline_res = load_precomputed_results(
                        baseline_dir, baseline_ts, split_name
                    )

                    if baseline_res:
                        py_list = baseline_res.get("pearson_per_channel", [])
                        pz_list = baseline_res.get("pearson_per_channel_Z", [])
                        if py_list:
                            baseline_r_y = np.array(py_list)
                        if pz_list:
                            baseline_r_z = np.array(pz_list)
                        baseline_yp = baseline_res.get("Yp")
                        baseline_zp = baseline_res.get("Zp")
                        baseline_name = variant_short_name(selected_baseline)

    sort_option = st.radio(
        "Sort trials by:",
        options=[
            "Original Order (Trial Index)",
            "Neural Mean Performance",
            "Behavioral Mean Performance",
        ],
        index=0,
        horizontal=True,
    )

    if sort_option == "Neural Mean Performance" and neural_available:
        df_y = df_y.sort_values("Mean_R", ascending=False)
        sort_order = df_y["Trial"].values
        if behavioral_available:
            df_z = df_z.set_index("Trial").loc[sort_order].reset_index()
    elif sort_option == "Behavioral Mean Performance" and behavioral_available:
        df_z = df_z.sort_values("Mean_R", ascending=False)
        sort_order = df_z["Trial"].values
        if neural_available:
            df_y = df_y.set_index("Trial").loc[sort_order].reset_index()
    else:
        pass

    def create_heatmap(df, channels, title, colorscale="Viridis"):
        y_labels = [f"Trial {i}" for i in df["Trial"]]

        fig = go.Figure(
            data=go.Heatmap(
                z=df[channels].values,
                x=channels,
                y=y_labels,
                colorscale=colorscale,
                colorbar=dict(title="Pearson r"),
                zmin=0,
                zmax=1,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Channels",
            yaxis_title="Trials",
            height=max(400, 300 + (len(df) * 20)),
            template="plotly_white",
        )
        return fig

    if neural_available:
        st.markdown("### Neural Prediction Performance (Pearson r)")

        render_normalized_kde_plot(
            r_y_arr,
            y_chan_names,
            split_res.get("block", []),
            "",
            use_bands=True,
            baseline_r_arr=baseline_r_y,
            baseline_name=baseline_name,
            model_name=model_name,
        )

        render_spectral_fvu_plot(
            split_res.get("Y", []),
            split_res.get("Yp", []),
            y_chan_names,
            "Neural Dynamics tracking (FVU)",
            sampling_freq=sampling_freq,
            use_bands=True,
            baseline_pred=baseline_yp,
            baseline_name=baseline_name,
            model_name=model_name,
        )

        render_distribution_comparison_plot(
            split_res.get("Y", []),
            split_res.get("Yp", []),
            y_chan_names,
            split_res.get("block", []),
            "",
            use_bands=True,
            baseline_pred=baseline_yp,
            baseline_name=baseline_name,
        )

        render_residual_acf_heatmap(
            split_res.get("Y", []),
            split_res.get("Yp", []),
            y_chan_names,
            "Neural Residuals Autocorrelation (ACF)",
            use_bands=True,
            baseline_pred=baseline_yp,
            baseline_name=baseline_name,
        )

        render_error_cdf_plot(
            split_res.get("Y", []),
            split_res.get("Yp", []),
            y_chan_names,
            "Neural Absolute Error CDF",
            use_bands=True,
            baseline_pred=baseline_yp,
            baseline_name=baseline_name,
        )

        render_residual_qq_plot(
            split_res.get("Y", []),
            split_res.get("Yp", []),
            y_chan_names,
            "Neural Residuals Normality (QQ Plot)",
            use_bands=True,
            baseline_pred=baseline_yp,
            baseline_name=baseline_name,
            model_name=model_name,
        )

        render_raincloud_plot(
            r_y_arr,
            y_chan_names,
            split_res.get("block", []),
            "",
            use_bands=True,
            baseline_r_arr=baseline_r_y,
            baseline_name=baseline_name,
        )

        fig_y = create_heatmap(
            df_y, y_chan_names, "Neural Performance Heatmap", "Plasma"
        )
        st.plotly_chart(fig_y, use_container_width=True)

    if behavioral_available:
        st.markdown("### Behavioral Prediction Performance (Pearson r)")

        render_normalized_kde_plot(
            r_z_arr,
            z_chan_names,
            split_res.get("block", []),
            "",
            use_bands=False,
            baseline_r_arr=baseline_r_z,
            baseline_name=baseline_name,
            model_name=model_name,
        )

        render_spectral_fvu_plot(
            split_res.get("Z", []),
            split_res.get("Zp", []),
            z_chan_names,
            "Behavioral Dynamics tracking (FVU)",
            sampling_freq=sampling_freq,
            use_bands=False,
            baseline_pred=baseline_zp,
            baseline_name=baseline_name,
            model_name=model_name,
        )

        render_distribution_comparison_plot(
            split_res.get("Z", []),
            split_res.get("Zp", []),
            z_chan_names,
            split_res.get("block", []),
            "",
            use_bands=False,
            baseline_pred=baseline_zp,
            baseline_name=baseline_name,
        )

        render_residual_acf_heatmap(
            split_res.get("Z", []),
            split_res.get("Zp", []),
            z_chan_names,
            "Behavioral Residuals Autocorrelation (ACF)",
            use_bands=False,
            baseline_pred=baseline_zp,
            baseline_name=baseline_name,
        )

        render_error_cdf_plot(
            split_res.get("Z", []),
            split_res.get("Zp", []),
            z_chan_names,
            "Behavioral Absolute Error CDF",
            use_bands=False,
            baseline_pred=baseline_zp,
            baseline_name=baseline_name,
        )

        render_residual_qq_plot(
            split_res.get("Z", []),
            split_res.get("Zp", []),
            z_chan_names,
            "Behavioral Residuals Normality (QQ Plot)",
            use_bands=False,
            baseline_pred=baseline_zp,
            baseline_name=baseline_name,
            model_name=model_name,
        )

        render_raincloud_plot(
            r_z_arr,
            z_chan_names,
            split_res.get("block", []),
            "",
            use_bands=False,
            baseline_r_arr=baseline_r_z,
            baseline_name=baseline_name,
        )

        fig_z = create_heatmap(
            df_z, z_chan_names, "Behavioral Performance Heatmap", "Viridis"
        )
        st.plotly_chart(fig_z, use_container_width=True)


def render_raincloud_plot(
    r_arr: np.ndarray,
    channel_names: list,
    blocks: list,
    title: str,
    use_bands: bool = False,
    baseline_r_arr: np.ndarray = None,
    baseline_name: str = "Baseline",
):
    st.markdown(f"#### {title}")

    if r_arr is None or len(channel_names) == 0:
        st.info("Insufficient data for Raincloud Plot.")
        return

    try:
        if r_arr.ndim != 2:
            st.warning("Data is not 2D.")
            return

        n_trials, n_chans = r_arr.shape

        active_channel_names = channel_names
        if len(active_channel_names) != n_chans:
            active_channel_names = [f"Ch{i}" for i in range(n_chans)]

        groups = _get_groups_map(active_channel_names, use_bands)

        if not blocks or len(blocks) != n_trials:
            blocks_clean = ["All Data"] * n_trials
        else:
            blocks_clean = [f"Block {b}" for b in blocks]

        data_list = []
        for t_idx in range(n_trials):
            block = blocks_clean[t_idx]
            for c_idx in range(n_chans):
                data_list.append(
                    {
                        "Block": block,
                        "Pearson r": r_arr[t_idx, c_idx],
                        "Group": groups[c_idx],
                        "Channel": active_channel_names[c_idx],
                    }
                )

            if baseline_r_arr is not None and len(baseline_r_arr) > t_idx:
                for c_idx in range(min(n_chans, baseline_r_arr.shape[1])):
                    data_list.append(
                        {
                            "Block": block,
                            "Pearson r": baseline_r_arr[t_idx, c_idx],
                            "Group": f"{groups[c_idx]} ({baseline_name})",
                            "Channel": active_channel_names[c_idx],
                        }
                    )

        df_plot = pd.DataFrame(data_list)

        def get_block_num(s):
            match = re.search(r"\d+", s)
            return int(match.group()) if match else 0

        unique_blocks = sorted(df_plot["Block"].unique(), key=get_block_num)
        df_plot["Block"] = pd.Categorical(
            df_plot["Block"], categories=unique_blocks, ordered=True
        )
        df_plot = df_plot.sort_values(["Block", "Group"])

        unique_groups = sorted(df_plot["Group"].unique())

        fig = go.Figure()

        colors = px.colors.qualitative.Plotly
        group_colors = {
            grp: colors[i % len(colors)] for i, grp in enumerate(unique_groups)
        }

        for grp in unique_groups:
            df_grp = df_plot[df_plot["Group"] == grp]
            color = group_colors[grp]

            fig.add_trace(
                go.Violin(
                    y=df_grp["Block"],
                    x=df_grp["Pearson r"],
                    name=grp,
                    legendgroup=grp,
                    showlegend=True,
                    line_color=color,
                    fillcolor=color,
                    opacity=0.3,
                    side="positive",
                    orientation="h",
                    width=1.0,
                    points=False,
                    meanline_visible=True,
                )
            )

        unique_blocks_reversed = list(reversed(unique_blocks))
        fig.update_layout(
            title=title,
            yaxis_title="Validation Block",
            xaxis_title="Pearson r",
            yaxis=dict(categoryorder="array", categoryarray=unique_blocks_reversed),
            xaxis=dict(zeroline=False, range=[-1.1, 1.1]),
            violinmode="group",
            template="plotly_white",
            height=max(600, 200 * len(unique_blocks)),
            hovermode="closest",
            legend=dict(title="Frequency Band" if use_bands else "Behavioral Feature"),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("Distribution of Pearson correlations across trials and channels.")

    except Exception as e:
        st.error(f"Error rendering Raincloud Plot: {e}")


def render_normalized_kde_plot(
    r_arr: np.ndarray,
    channel_names: list,
    blocks: list,
    title: str,
    use_bands: bool = False,
    baseline_r_arr: np.ndarray = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
):
    if r_arr is None or len(channel_names) == 0:
        return

    n_trials, n_chans = r_arr.shape

    groups = _get_groups_map(channel_names, use_bands)
    unique_groups = sorted(list(set(groups)))

    blocks_arr = (
        np.array(blocks) if (blocks and len(blocks) == n_trials) else np.zeros(n_trials)
    )
    unique_blks = np.unique(blocks_arr)

    group_stats = {}
    for grp in unique_groups:
        grp_indices = [idx for idx, g in enumerate(groups) if g == grp]
        grp_data = r_arr[:, grp_indices].flatten()

        mean = np.mean(grp_data)
        std = np.std(grp_data)
        sem = std / np.sqrt(len(grp_data)) if len(grp_data) > 1 else 0

        group_stats[grp] = {
            "mean": mean,
            "std": std,
            "sem": sem,
            "n": len(grp_data),
        }

        if baseline_r_arr is not None:
            main_matched_idx, base_matched_idx = _match_channels_by_name(
                channel_names, channel_names
            )
            valid_grp_main = [idx for idx in grp_indices if idx in main_matched_idx]
            valid_grp_base = [
                base_matched_idx[main_matched_idx.index(idx)]
                for idx in valid_grp_main
                if idx in main_matched_idx
            ]
            valid_grp_base = [
                idx for idx in valid_grp_base if idx < baseline_r_arr.shape[1]
            ]

            if valid_grp_base:
                base_grp_data = baseline_r_arr[:, valid_grp_base].flatten()
                base_mean = np.mean(base_grp_data)
                base_std = np.std(base_grp_data)
                base_sem = (
                    base_std / np.sqrt(len(base_grp_data))
                    if len(base_grp_data) > 1
                    else 0
                )

                group_stats[grp]["baseline_mean"] = base_mean
                group_stats[grp]["baseline_std"] = base_std
                group_stats[grp]["baseline_sem"] = base_sem
                group_stats[grp]["baseline_n"] = len(base_grp_data)
                group_stats[grp]["improvement"] = mean - base_mean
                group_stats[grp]["improvement_pct"] = (
                    ((mean - base_mean) / abs(base_mean) * 100)
                    if abs(base_mean) > 1e-6
                    else 0
                )

    plot_data = []

    for grp in unique_groups:
        grp_indices = [idx for idx, g in enumerate(groups) if g == grp]
        model_data = r_arr[:, grp_indices].flatten()

        # Add model data
        for val in model_data:
            plot_data.append(
                {
                    "Feature": grp,
                    "Model": model_name,
                    "Pearson r": val,
                }
            )

        # Add baseline data if available
        if baseline_r_arr is not None:
            # Match channels by name
            main_matched_idx, base_matched_idx = _match_channels_by_name(
                channel_names, channel_names
            )
            valid_grp_main = [idx for idx in grp_indices if idx in main_matched_idx]
            valid_grp_base = [
                base_matched_idx[main_matched_idx.index(idx)]
                for idx in valid_grp_main
                if idx in main_matched_idx
            ]
            valid_grp_base = [
                idx for idx in valid_grp_base if idx < baseline_r_arr.shape[1]
            ]

            if valid_grp_base:
                baseline_data = baseline_r_arr[:, valid_grp_base].flatten()
                for val in baseline_data:
                    plot_data.append(
                        {
                            "Feature": grp,
                            "Model": baseline_name,
                            "Pearson r": val,
                        }
                    )

    df_plot = pd.DataFrame(plot_data)

    feature_to_pos = {grp: i for i, grp in enumerate(unique_groups)}
    df_plot["x_pos"] = df_plot["Feature"].map(feature_to_pos)

    # Create base figure using backbone utility
    fig = create_base_comparison_figure(
        x_label="Frequency Band" if use_bands else "Feature",
        y_label="Pearson r",
        title=title,
    )

    # Add box plots and strip plots for each model type
    for model_type in [model_name, baseline_name]:
        if model_type == baseline_name and baseline_r_arr is None:
            continue

        df_model = df_plot[df_plot["Model"] == model_type]

        if len(df_model) == 0:
            continue

        fig.add_trace(
            go.Box(
                x=df_model["x_pos"],
                y=df_model["Pearson r"],
                name=model_type,
                boxmean="sd",
                marker_color=(
                    PALETTE.twilight_indigo
                    if model_type == model_name
                    else BASELINE_COLOR
                ),
                line_color=(
                    PALETTE.twilight_indigo
                    if model_type == model_name
                    else BASELINE_COLOR
                ),
                fillcolor=(
                    PALETTE.twilight_indigo
                    if model_type == model_name
                    else BASELINE_COLOR
                ),
                opacity=0.6,
                showlegend=True,
                legendgroup=model_type,
            )
        )

        for grp in unique_groups:
            grp_data = df_model[df_model["Feature"] == grp]["Pearson r"].values
            if len(grp_data) > 0:
                x_numeric = feature_to_pos[grp]
                if model_type == model_name:
                    x_jitter = np.random.normal(-0.15, 0.08, len(grp_data))
                else:
                    x_jitter = np.random.normal(0.15, 0.08, len(grp_data))
                x_pos = x_numeric + x_jitter

                fig.add_trace(
                    go.Scatter(
                        x=x_pos,
                        y=grp_data,
                        mode="markers",
                        name=model_type,
                        marker=dict(
                            color=(
                                PALETTE.twilight_indigo
                                if model_type == model_name
                                else BASELINE_COLOR
                            ),
                            size=4,
                            opacity=0.4,
                            line=dict(width=0.5, color="white"),
                        ),
                        showlegend=False,
                        legendgroup=model_type,
                        hoverinfo="y",
                    )
                )

    fig.update_layout(
        height=500,
        boxmode="group",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(unique_groups))),
            ticktext=unique_groups,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics table
    if baseline_r_arr is not None:
        st.markdown("##### Performance Summary")
        summary_data = []
        for grp in unique_groups:
            stats = group_stats[grp]
            if "baseline_mean" in stats:
                summary_data.append(
                    {
                        "Feature": grp,
                        f"{baseline_name} Mean": f"{stats['baseline_mean']:.4f}",
                        f"{baseline_name} SEM": f"{stats['baseline_sem']:.4f}",
                        "Model Mean": f"{stats['mean']:.4f}",
                        "Model SEM": f"{stats['sem']:.4f}",
                        "Improvement": f"{stats['improvement']:+.4f}",
                        "Improvement %": f"{stats['improvement_pct']:+.2f}%",
                    }
                )

        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

    st.caption(
        "Distribution of Pearson correlations across all trials and channels. Box plots show median, quartiles, and outliers."
    )


def render_distribution_comparison_plot(
    data_true: list,
    data_pred: list,
    channel_names: list,
    blocks: list,
    title: str,
    use_bands: bool = False,
    baseline_pred: list = None,
    baseline_name: str = "Baseline",
):
    if not data_true or not data_pred or len(channel_names) == 0:
        return

    n_trials = len(data_true)
    blocks_arr = (
        np.array(blocks) if (blocks and len(blocks) == n_trials) else np.zeros(n_trials)
    )
    unique_blks = np.unique(blocks_arr)

    n_chans = len(channel_names)
    groups_map = _get_groups_map(channel_names, use_bands)

    unique_groups = sorted(list(set(groups_map)))
    n_groups = len(unique_groups)

    # Calculate rows/cols for grid
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"<b>{grp}</b>" for grp in unique_groups],
        shared_xaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    x_range = np.linspace(-3.5, 4.5, 200)

    _ks_captions = []  # Collect KS stats for caption below chart
    for i, grp in enumerate(unique_groups):
        grp_indices = [idx for idx, g in enumerate(groups_map) if g == grp]

        pooled_true = []
        pooled_pred = []
        pooled_base = []

        for blk in unique_blks:
            blk_indices = np.where(blocks_arr == blk)[0]

            blk_true_samples = []
            blk_pred_samples = []
            blk_base_samples = []

            for t_idx in blk_indices:
                t_true = np.array(data_true[t_idx])
                t_pred = np.array(data_pred[t_idx])
                t_base = (
                    np.array(baseline_pred[t_idx])
                    if baseline_pred
                    and t_idx < len(baseline_pred)
                    and baseline_pred[t_idx] is not None
                    else None
                )

                if t_true.ndim == 2:
                    blk_true_samples.append(t_true[:, grp_indices])
                    blk_pred_samples.append(t_pred[:, grp_indices])
                    if t_base is not None:
                        main_matched_idx, base_matched_idx = _match_channels_by_name(
                            channel_names, channel_names
                        )
                        valid_grp_main = [
                            idx for idx in grp_indices if idx in main_matched_idx
                        ]
                        valid_grp_base = [
                            base_matched_idx[main_matched_idx.index(idx)]
                            for idx in valid_grp_main
                            if idx in main_matched_idx
                        ]
                        valid_grp_base = [
                            idx for idx in valid_grp_base if idx < t_base.shape[1]
                        ]
                        if valid_grp_base:
                            blk_base_samples.append(t_base[:, valid_grp_base])
                else:
                    blk_true_samples.append(t_true[grp_indices])
                    blk_pred_samples.append(t_pred[grp_indices])
                    if t_base is not None:
                        # Match channels by name
                        main_matched_idx, base_matched_idx = _match_channels_by_name(
                            channel_names, channel_names
                        )
                        valid_grp_main = [
                            idx for idx in grp_indices if idx in main_matched_idx
                        ]
                        valid_grp_base = [
                            base_matched_idx[main_matched_idx.index(idx)]
                            for idx in valid_grp_main
                            if idx in main_matched_idx
                        ]
                        valid_grp_base = [
                            idx for idx in valid_grp_base if idx < len(t_base)
                        ]
                        if valid_grp_base:
                            blk_base_samples.append(t_base[valid_grp_base])

            if not blk_true_samples:
                continue

            blk_true_flat = np.vstack(
                [np.atleast_2d(s) if s.ndim == 1 else s for s in blk_true_samples]
            )
            blk_pred_flat = np.vstack(
                [np.atleast_2d(s) if s.ndim == 1 else s for s in blk_pred_samples]
            )

            blk_base_flat = np.empty((0,))
            if blk_base_samples:
                blk_base_flat = np.vstack(
                    [np.atleast_2d(s) if s.ndim == 1 else s for s in blk_base_samples]
                )

            mean = np.mean(blk_true_flat)
            std = np.std(blk_true_flat)

            if std > 1e-6:
                pooled_true.append(((blk_true_flat - mean) / std).flatten())
                pooled_pred.append(((blk_pred_flat - mean) / std).flatten())
                if len(blk_base_flat) > 0:
                    pooled_base.append(((blk_base_flat - mean) / std).flatten())
            else:
                pooled_true.append((blk_true_flat - mean).flatten())
                pooled_pred.append((blk_pred_flat - mean).flatten())
                if len(blk_base_flat) > 0:
                    pooled_base.append((blk_base_flat - mean).flatten())

        if not pooled_true:
            continue

        final_true = np.concatenate(pooled_true)
        final_pred = np.concatenate(pooled_pred)
        final_base = np.concatenate(pooled_base) if pooled_base else np.array([])

        # KS Test per panel
        ks_stat, p_val = ks_2samp(final_true, final_pred)
        ks_stat_base = 0
        if len(final_base) > 0:
            ks_stat_base, _ = ks_2samp(final_true, final_base)

        # Subsample for speed
        max_pts = 5000
        if len(final_true) > max_pts:
            idx = np.random.choice(len(final_true), max_pts, replace=False)
            t_kde_data = final_true[idx]
        else:
            t_kde_data = final_true

        if len(final_pred) > max_pts:
            idx = np.random.choice(len(final_pred), max_pts, replace=False)
            p_kde_data = final_pred[idx]
        else:
            p_kde_data = final_pred

        b_kde_data = None
        if len(final_base) > 0:
            if len(final_base) > max_pts:
                idx = np.random.choice(len(final_base), max_pts, replace=False)
                b_kde_data = final_base[idx]
            else:
                b_kde_data = final_base

        kde_true = gaussian_kde(t_kde_data)(x_range)
        kde_pred = gaussian_kde(p_kde_data)(x_range)
        kde_base = gaussian_kde(b_kde_data)(x_range) if b_kde_data is not None else None

        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

        # True (Shadow)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_true,
                fill="tozeroy",
                name="True Data",
                line_color="rgba(0,0,0,0.3)",
                fillcolor="rgba(0,0,0,0.1)",
                showlegend=(i == 0),
                legendgroup="True",
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Predicted (Dashed Red)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_pred,
                name="Prediction",
                line=dict(color="#D50032", width=2, dash="dash"),
                showlegend=(i == 0),
                legendgroup="Pred",
                hoverinfo="x+y",
            ),
            row=row,
            col=col,
        )

        if kde_base is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde_base,
                    name=baseline_name,
                    line=dict(color="#00E5FF", width=2, dash="dot"),
                    showlegend=(i == 0),
                    legendgroup="Baseline",
                    hoverinfo="x+y",
                ),
                row=row,
                col=col,
            )

        # Collect KS stats for caption
        ks_parts = [f"{grp}: Pred {ks_stat:.3f}"]
        if len(final_base) > 0:
            ks_parts[0] += f", Base {ks_stat_base:.3f}"
        _ks_captions.append(ks_parts[0])

    fig.update_layout(
        title=f"Distribution Match by {'Band' if use_bands else 'Feature'}",
        template="plotly_white",
        height=300 * n_rows,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Amplitude (Z-scored)", row=n_rows, col=2)
    fig.update_yaxes(title_text="Density", col=1)

    st.plotly_chart(fig, use_container_width=True)
    if _ks_captions:
        st.caption("KS distance (Data↔Model) — " + " | ".join(_ks_captions))


def render_spectral_fvu_plot(
    data_true: list,
    data_pred: list,
    channel_names: list,
    title: str,
    sampling_freq: float = 80.0,
    use_bands: bool = False,
    baseline_pred: list = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
):
    if not data_true or not data_pred or len(channel_names) == 0:
        return

    # 1. Filter and identify groups
    n_chans = len(channel_names)
    groups_map = _get_groups_map(channel_names, use_bands)

    unique_groups = sorted(list(set(groups_map)))
    n_groups = len(unique_groups)

    # 2. Concatenate all trials for frequency analysis
    # trials are list of [Time, Channels]
    try:
        y_true_all = np.concatenate([np.atleast_2d(t) for t in data_true], axis=0)
        y_pred_all = np.concatenate([np.atleast_2d(t) for t in data_pred], axis=0)
        y_base_all = (
            np.concatenate(
                [np.atleast_2d(t) for t in baseline_pred if t is not None], axis=0
            )
            if baseline_pred
            else None
        )
    except:
        return

    if y_true_all.shape[1] != n_chans or y_pred_all.shape[1] != n_chans:
        return

    # 3. Create Grid
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"<b>{grp}</b>" for grp in unique_groups],
        shared_xaxes=False,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    for i, grp in enumerate(unique_groups):
        grp_indices = [idx for idx, g in enumerate(groups_map) if g == grp]

        # Extract band data
        true_grp = y_true_all[:, grp_indices]
        pred_grp = y_pred_all[:, grp_indices]
        resid_grp = true_grp - pred_grp

        # Compute PSDs across channels in group
        # nperseg should be reasonable for the modulation freq.
        # Using 256 for 60Hz gives ~0.23Hz resolution.
        nperseg = min(len(true_grp), 256)
        f, P_true = welch(true_grp, fs=sampling_freq, nperseg=nperseg, axis=0)
        f, P_resid = welch(resid_grp, fs=sampling_freq, nperseg=nperseg, axis=0)

        P_base_resid = None
        P_true_for_base = None
        if y_base_all is not None:
            # Match channels by name (channels have same names in both models)
            # Since channels have same names, match indices from main to baseline
            main_matched_idx, base_matched_idx = _match_channels_by_name(
                channel_names, channel_names  # Same names, but verify matching
            )
            # Filter to only include channels in the current group
            valid_main_idx = [idx for idx in main_matched_idx if idx in grp_indices]
            valid_base_idx = [
                base_matched_idx[main_matched_idx.index(idx)]
                for idx in valid_main_idx
                if idx in main_matched_idx
            ]

            # Ensure baseline has enough channels
            valid_base_idx = [
                idx for idx in valid_base_idx if idx < y_base_all.shape[1]
            ]
            valid_main_idx = [
                main_matched_idx[base_matched_idx.index(idx)]
                for idx in valid_base_idx
                if idx in base_matched_idx
            ]

            if valid_main_idx and valid_base_idx:
                base_grp = y_base_all[:, valid_base_idx]
                true_grp_matched = y_true_all[:, valid_main_idx]
                min_len = min(len(true_grp_matched), len(base_grp))
                _P_true_m = true_grp_matched[:min_len]
                _P_base = base_grp[:min_len]
                f, P_base_resid = welch(
                    _P_true_m - _P_base, fs=sampling_freq, nperseg=nperseg, axis=0
                )
                # Compute P_true on the same matched channels for correct ratio
                _, P_true_for_base = welch(
                    _P_true_m, fs=sampling_freq, nperseg=nperseg, axis=0
                )

        # FVU Ratio per channel
        # P_true is (Freq, ChansInGroup)
        ratios = P_resid / (P_true + 1e-10)

        # Stats across channels
        mean_ratio = np.mean(ratios, axis=1)
        sem_ratio = np.std(ratios, axis=1) / np.sqrt(max(1, ratios.shape[1]))

        mean_base_ratio, sem_base_ratio = None, None
        if P_base_resid is not None and P_true_for_base is not None:
            base_ratios = P_base_resid / (P_true_for_base + 1e-10)
            mean_base_ratio = np.mean(base_ratios, axis=1)
            sem_base_ratio = np.std(base_ratios, axis=1) / np.sqrt(
                max(1, base_ratios.shape[1])
            )

        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

        # Plot Mean Line (model)
        fig.add_trace(
            go.Scatter(
                x=f,
                y=mean_ratio,
                line=dict(color="black", width=2),
                name=model_name,
                showlegend=(i == 0),
                legendgroup="Model",
                hoverinfo="x+y",
            ),
            row=row,
            col=col,
        )

        # Plot SEM Shading
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([f, f[::-1]]),
                y=np.concatenate(
                    [mean_ratio + sem_ratio, (mean_ratio - sem_ratio)[::-1]]
                ),
                fill="toself",
                fillcolor="rgba(128, 128, 128, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"{grp} SEM",
            ),
            row=row,
            col=col,
        )

        if mean_base_ratio is not None:
            fig.add_trace(
                go.Scatter(
                    x=f,
                    y=mean_base_ratio,
                    line=dict(color="#00E5FF", width=2, dash="dash"),
                    name=baseline_name,
                    showlegend=(i == 0),
                    legendgroup="Baseline",
                    hoverinfo="x+y",
                ),
                row=row,
                col=col,
            )

        # Chance line at FVU=1 (model explains nothing)
        fig.add_trace(
            go.Scatter(
                x=[0, max(f)],
                y=[1, 1],
                mode="lines",
                line=dict(color="red", width=1, dash="dash"),
                name="Chance (FVU=1)",
                showlegend=(i == 0),
                legendgroup="Chance",
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Zoom to interesting modulation frequencies (< 10Hz)
        fig.update_xaxes(range=[0, min(10, sampling_freq / 2)], row=row, col=col)

    fig.update_layout(
        title=f"Spectral Fidelity (FVU) by {'Band' if use_bands else 'Feature'}",
        template="plotly_white",
        height=300 * n_rows,
        yaxis=dict(range=[0, 1.2]),
    )
    fig.update_xaxes(title_text="Modulation Freq (Hz)", row=n_rows, col=2)
    fig.update_yaxes(title_text="Unexplained Var. Ratio", col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_residual_acf_heatmap(
    data_true: list,
    data_pred: list,
    channel_names: list,
    title: str,
    use_bands: bool = False,
    baseline_pred: list = None,
    baseline_name: str = "Baseline",
):
    if not data_true or not data_pred or not channel_names:
        return
    try:
        y_true = np.concatenate([np.atleast_2d(t) for t in data_true], axis=0)
        y_pred = np.concatenate([np.atleast_2d(t) for t in data_pred], axis=0)
        y_base = (
            np.concatenate(
                [np.atleast_2d(t) for t in baseline_pred if t is not None], axis=0
            )
            if baseline_pred
            else None
        )

        residuals = y_true - y_pred
        b_residuals = None
        if y_base is not None:
            # handle potential length AND channel mismatches
            min_l = min(y_true.shape[0], y_base.shape[0])
            min_c = min(y_true.shape[1], y_base.shape[1])
            b_residuals = y_true[:min_l, :min_c] - y_base[:min_l, :min_c]

        nlags = 20
        n_chans = len(channel_names)
        n_base_chans = y_base.shape[1] if y_base is not None else 0
        n_rows_acf = (
            n_chans + min(n_chans, n_base_chans) if y_base is not None else n_chans
        )
        acf_matrix = np.zeros((n_rows_acf, nlags + 1))

        for c in range(n_chans):
            acf_matrix[c, :] = acf(residuals[:, c], nlags=nlags, fft=True)
            if b_residuals is not None and c < n_base_chans:
                acf_matrix[n_chans + c, :] = acf(
                    b_residuals[:, c], nlags=nlags, fft=True
                )

        y_labels = _get_groups_map(channel_names, use_bands)

        all_y_labels = y_labels.copy()
        if y_base is not None:
            all_y_labels.extend(
                [
                    f"{y_labels[c]} ({baseline_name})"
                    for c in range(min(n_chans, n_base_chans))
                ]
            )

        sort_idx = np.argsort(all_y_labels)
        acf_matrix = acf_matrix[: len(all_y_labels)][sort_idx, :]
        sorted_y_labels = [all_y_labels[i] for i in sort_idx]

        fig = go.Figure(
            data=go.Heatmap(
                z=acf_matrix[:, 1:],  # Skip lag 0
                x=list(range(1, nlags + 1)),
                y=sorted_y_labels,
                colorscale="RdBu",
                zmid=0,
                zmax=1,
                zmin=-1,
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Lag (timesteps)",
            yaxis_title="Channels" if y_base is None else "",
            height=max(500, 20 * len(all_y_labels)),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"ACF Error: {e}")


def render_error_cdf_plot(
    data_true: list,
    data_pred: list,
    channel_names: list,
    title: str,
    use_bands: bool = False,
    baseline_pred: list = None,
    baseline_name: str = "Baseline",
):
    """
    Empirical CDF of absolute prediction errors per channel group.

    Computation:
      1. Concatenate all trials into (TotalSamples, Channels) arrays.
      2. Compute per-sample absolute error: |y_true − y_pred|.
      3. Group channels by frequency band (use_bands) or keep raw names.
      4. Plot the empirical CDF via plotly express ``ecdf``.

    Interpretation: a CDF curve shifted further left indicates smaller
    errors and therefore a better-fitting model.
    """
    if not data_true or not data_pred or not channel_names:
        return
    try:
        y_true = np.concatenate([np.atleast_2d(t) for t in data_true], axis=0)
        y_pred = np.concatenate([np.atleast_2d(t) for t in data_pred], axis=0)
        y_base = (
            np.concatenate(
                [np.atleast_2d(t) for t in baseline_pred if t is not None], axis=0
            )
            if baseline_pred
            else None
        )

        abs_err = np.abs(y_true - y_pred)
        b_abs_err = None
        channel_to_base_idx = {}
        if y_base is not None:
            # Match channels by name (channels have same names, so 1:1 mapping)
            main_matched_idx, base_matched_idx = _match_channels_by_name(
                channel_names, channel_names
            )
            min_l = min(y_true.shape[0], y_base.shape[0])
            valid_base_idx = [idx for idx in base_matched_idx if idx < y_base.shape[1]]
            valid_main_idx = [
                main_matched_idx[base_matched_idx.index(idx)]
                for idx in valid_base_idx
                if idx in base_matched_idx
            ]
            if valid_main_idx and valid_base_idx:
                b_abs_err = np.abs(
                    y_true[:min_l, valid_main_idx] - y_base[:min_l, valid_base_idx]
                )
                # Create mapping for error CDF grouping
                channel_to_base_idx = {
                    valid_main_idx[i]: valid_base_idx[i]
                    for i in range(len(valid_main_idx))
                }

        groups = _get_groups_map(channel_names, use_bands)

        df_list = []

        for c in range(len(channel_names)):
            # Subsample for plot performance
            err_vals = abs_err[:, c]
            if len(err_vals) > 1000:
                err_vals = np.random.choice(err_vals, 1000, replace=False)
            for v in err_vals:
                df_list.append({"Abs Error": v, "Group": groups[c]})

            if b_abs_err is not None and c in channel_to_base_idx:
                base_c = channel_to_base_idx[c]
                if base_c < b_abs_err.shape[1]:
                    b_err_vals = b_abs_err[:, base_c]
                    if len(b_err_vals) > 1000:
                        b_err_vals = np.random.choice(b_err_vals, 1000, replace=False)
                    for v in b_err_vals:
                        df_list.append(
                            {"Abs Error": v, "Group": f"{groups[c]} ({baseline_name})"}
                        )

        df = pd.DataFrame(df_list)
        fig = px.ecdf(
            df, x="Abs Error", color="Group", title=title, template="plotly_white"
        )
        fig.update_layout(
            xaxis_title="Absolute Error |y_true - y_pred|",
            yaxis_title="Cumulative Probability",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"CDF Error: {e}")


def render_residual_qq_plot(
    data_true: list,
    data_pred: list,
    channel_names: list,
    title: str,
    use_bands: bool = False,
    baseline_pred: list = None,
    baseline_name: str = "Baseline",
    model_name: str = "Model",
):
    """
    QQ plot of standardised residuals against a normal distribution.

    Computation:
      1. Residuals = y_true − y_pred, pooled across trials per group.
      2. Standardised to mean=0, std=1.
      3. ``probplot`` computes theoretical vs sample quantiles.

    Interpretation: points on the red dashed diagonal indicate
    perfectly Gaussian residuals.  Heavy tails (deviations at the
    ends) reveal the model struggles with extreme values / bursts.
    """
    if not data_true or not data_pred or not channel_names:
        return
    try:
        y_true = np.concatenate([np.atleast_2d(t) for t in data_true], axis=0)
        y_pred = np.concatenate([np.atleast_2d(t) for t in data_pred], axis=0)
        resids = y_true - y_pred

        y_base = None
        if baseline_pred:
            y_base = np.concatenate(
                [np.atleast_2d(t) for t in baseline_pred if t is not None], axis=0
            )

        groups_map = _get_groups_map(channel_names, use_bands)

        unique_groups = sorted(list(set(groups_map)))
        n_cols = 3
        n_rows = (len(unique_groups) + n_cols - 1) // n_cols
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=unique_groups)

        _baseline_legend_shown = False
        for i, grp in enumerate(unique_groups):
            grp_idx = [idx for idx, g in enumerate(groups_map) if g == grp]
            grp_resids = resids[:, grp_idx].flatten()
            grp_resids = (grp_resids - np.mean(grp_resids)) / (
                np.std(grp_resids) + 1e-8
            )

            if len(grp_resids) > 2000:
                grp_resids = np.random.choice(grp_resids, 2000, replace=False)

            osm, osr = probplot(grp_resids, dist="norm")[0]

            row, col = (i // n_cols) + 1, (i % n_cols) + 1

            # Model residuals
            fig.add_trace(
                go.Scatter(
                    x=osm,
                    y=osr,
                    mode="markers",
                    marker=dict(size=3, color="black", opacity=0.5),
                    name=model_name,
                    legendgroup="Model",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

            # Baseline residuals
            if y_base is not None:
                # Match channels by name
                main_matched_idx, base_matched_idx = _match_channels_by_name(
                    channel_names, channel_names
                )
                valid_grp_main = [idx for idx in grp_idx if idx in main_matched_idx]
                valid_grp_base = [
                    base_matched_idx[main_matched_idx.index(idx)]
                    for idx in valid_grp_main
                    if idx in main_matched_idx
                ]
                valid_grp_base = [
                    idx for idx in valid_grp_base if idx < y_base.shape[1]
                ]
                valid_grp_main = [
                    main_matched_idx[base_matched_idx.index(idx)]
                    for idx in valid_grp_base
                    if idx in base_matched_idx
                ]

                if valid_grp_main and valid_grp_base:
                    min_len = min(y_true.shape[0], y_base.shape[0])
                    base_resids = (
                        y_true[:min_len, valid_grp_main]
                        - y_base[:min_len, valid_grp_base]
                    ).flatten()
                    base_resids = (base_resids - np.mean(base_resids)) / (
                        np.std(base_resids) + 1e-8
                    )
                    if len(base_resids) > 2000:
                        base_resids = np.random.choice(base_resids, 2000, replace=False)
                    osm_b, osr_b = probplot(base_resids, dist="norm")[0]
                    fig.add_trace(
                        go.Scatter(
                            x=osm_b,
                            y=osr_b,
                            mode="markers",
                            marker=dict(size=3, color="#00E5FF", opacity=0.5),
                            name=baseline_name,
                            legendgroup="Baseline",
                            showlegend=(not _baseline_legend_shown),
                        ),
                        row=row,
                        col=col,
                    )
                    _baseline_legend_shown = True

            # Reference diagonal
            fig.add_trace(
                go.Scatter(
                    x=[-3, 3],
                    y=[-3, 3],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="Normal",
                    legendgroup="Normal",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

        fig.update_layout(title=title, height=300 * n_rows, template="plotly_white")
        fig.update_xaxes(title_text="Theoretical Quantiles")
        fig.update_yaxes(title_text="Sample Quantiles")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"QQ Error: {e}")
