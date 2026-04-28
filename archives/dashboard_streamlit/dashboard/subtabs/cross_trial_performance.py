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


def _render_performance_tab(
    split_res: Dict[str, Any],
    sampling_freq: float = 80.0,
    cfg_path: Any = None,
    y_true_key: str = "Y",
    y_pred_key: str = "Yp",
    z_true_key: str = "Z",
    z_pred_key: str = "Zp",
    title_prefix: str = "",
    baseline_key_prefix: str = "",
):
    """Internal function to render performance tab for either predictions or forecasts."""
    title = (
        f"{title_prefix}Performance Overview Across All Trials"
        if title_prefix
        else "Performance Overview Across All Trials"
    )
    st.subheader(title)

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

        key_suffix = (
            baseline_key_prefix.replace(" ", "_").lower()
            if baseline_key_prefix
            else "global"
        )

        if baseline_variants:
            selected_baseline = st.selectbox(
                "Select Baseline Model",
                options=["None"] + baseline_variants,
                index=1 if baseline_variants else 0,
                key=f"{key_suffix}_baseline_select",
            )

            if selected_baseline != "None":
                baseline_dir = results_root / selected_baseline
                baseline_timestamps = list_run_timestamps(baseline_dir)

                if baseline_timestamps:
                    baseline_ts = st.selectbox(
                        "Baseline run timestamp",
                        options=baseline_timestamps,
                        index=len(baseline_timestamps) - 1,
                        key=f"{key_suffix}_baseline_ts_select",
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
                        baseline_yp = baseline_res.get(y_pred_key)
                        baseline_zp = baseline_res.get(z_pred_key)
                        baseline_name = variant_short_name(selected_baseline)

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

    perf_type = "Forecast" if "future" in y_pred_key.lower() else "Prediction"

    if neural_available:
        st.markdown(f"### Neural {perf_type} Performance (Pearson r)")

        render_normalized_kde_plot(
            r_y_arr,
            y_chan_names,
            split_res.get("block", []),
            "",
            use_bands=True,
            baseline_r_arr=baseline_r_y,
            baseline_name=baseline_name,
            model_name=model_name,
            key_suffix=f"{key_suffix}_neural_kde",
        )

        render_residual_acf_heatmap(
            split_res.get(y_true_key, []),
            split_res.get(y_pred_key, []),
            y_chan_names,
            f"Neural {perf_type} Residuals Autocorrelation (ACF)",
            use_bands=True,
            baseline_pred=baseline_yp,
            baseline_name=baseline_name,
            key_suffix=f"{key_suffix}_neural_acf",
        )

        render_raincloud_plot(
            r_y_arr,
            y_chan_names,
            split_res.get("block", []),
            "",
            use_bands=True,
            baseline_r_arr=baseline_r_y,
            baseline_name=baseline_name,
            key_suffix=f"{key_suffix}_neural_raincloud",
        )

        fig_y = create_heatmap(
            df_y, y_chan_names, f"Neural {perf_type} Performance Heatmap", "Plasma"
        )
        st.plotly_chart(
            fig_y, use_container_width=True, key=f"{key_suffix}_neural_heatmap"
        )

    if behavioral_available:
        st.markdown(f"### Behavioral {perf_type} Performance (Pearson r)")

        render_normalized_kde_plot(
            r_z_arr,
            z_chan_names,
            split_res.get("block", []),
            "",
            use_bands=False,
            baseline_r_arr=baseline_r_z,
            baseline_name=baseline_name,
            model_name=model_name,
            key_suffix=f"{key_suffix}_behavioral_kde",
        )

        render_residual_acf_heatmap(
            split_res.get(z_true_key, []),
            split_res.get(z_pred_key, []),
            z_chan_names,
            f"Behavioral {perf_type} Residuals Autocorrelation (ACF)",
            use_bands=False,
            baseline_pred=baseline_zp,
            baseline_name=baseline_name,
            key_suffix=f"{key_suffix}_behavioral_acf",
        )

        render_raincloud_plot(
            r_z_arr,
            z_chan_names,
            split_res.get("block", []),
            "",
            use_bands=False,
            baseline_r_arr=baseline_r_z,
            baseline_name=baseline_name,
            key_suffix=f"{key_suffix}_behavioral_raincloud",
        )

        fig_z = create_heatmap(
            df_z, z_chan_names, f"Behavioral {perf_type} Performance Heatmap", "Viridis"
        )
        st.plotly_chart(
            fig_z, use_container_width=True, key=f"{key_suffix}_behavioral_heatmap"
        )


def render_cross_trial_performance_tab(
    split_res: Dict[str, Any], sampling_freq: float = 80.0, cfg_path: Any = None
):
    _render_performance_tab(
        split_res,
        sampling_freq=sampling_freq,
        cfg_path=cfg_path,
        y_true_key="Y",
        y_pred_key="Yp",
        z_true_key="Z",
        z_pred_key="Zp",
        baseline_key_prefix="predictions",
    )


def render_forecast_performance_tab(
    split_res: Dict[str, Any], sampling_freq: float = 80.0, cfg_path: Any = None
):
    _render_performance_tab(
        split_res,
        sampling_freq=sampling_freq,
        cfg_path=cfg_path,
        y_true_key="Y_future_true",
        y_pred_key="Y_future_pred",
        z_true_key="Z_future_true",
        z_pred_key="Z_future_pred",
        title_prefix="Forecast ",
        baseline_key_prefix="forecasts",
    )


def render_raincloud_plot(
    r_arr: np.ndarray,
    channel_names: list,
    blocks: list,
    title: str,
    use_bands: bool = False,
    baseline_r_arr: np.ndarray = None,
    baseline_name: str = "Baseline",
    key_suffix: str = "raincloud",
):
    st.markdown(f"#### {title}")

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

        st.plotly_chart(fig, use_container_width=True, key=f"{key_suffix}_chart")
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
    key_suffix: str = "kde",
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

    st.plotly_chart(fig, use_container_width=True, key=f"{key_suffix}_chart")

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


def render_residual_acf_heatmap(
    data_true: list,
    data_pred: list,
    channel_names: list,
    title: str,
    use_bands: bool = False,
    baseline_pred: list = None,
    baseline_name: str = "Baseline",
    key_suffix: str = "acf",
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
        st.plotly_chart(fig, use_container_width=True, key=f"{key_suffix}_chart")
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
