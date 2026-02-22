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


def render_cross_trial_performance_tab(
    split_res: Dict[str, Any], sampling_freq: float = 80.0
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
            r_y_arr, y_chan_names, split_res.get("block", []), "", use_bands=True
        )

        render_spectral_fvu_plot(
            split_res.get("Y", []),
            split_res.get("Yp", []),
            y_chan_names,
            "Neural Dynamics tracking (FVU)",
            sampling_freq=sampling_freq,
            use_bands=True,
        )

        render_distribution_comparison_plot(
            split_res.get("Y", []),
            split_res.get("Yp", []),
            y_chan_names,
            split_res.get("block", []),
            "",
            use_bands=True,
        )

        render_residual_acf_heatmap(
            split_res.get("Y", []),
            split_res.get("Yp", []),
            y_chan_names,
            "Neural Residuals Autocorrelation (ACF)",
            use_bands=True,
        )

        render_error_cdf_plot(
            split_res.get("Y", []),
            split_res.get("Yp", []),
            y_chan_names,
            "Neural Absolute Error CDF",
            use_bands=True,
        )

        render_residual_qq_plot(
            split_res.get("Y", []),
            split_res.get("Yp", []),
            y_chan_names,
            "Neural Residuals Normality (QQ Plot)",
            use_bands=True,
        )

        render_raincloud_plot(
            r_y_arr, y_chan_names, split_res.get("block", []), "", use_bands=True
        )

        fig_y = create_heatmap(
            df_y, y_chan_names, "Neural Performance Heatmap", "Plasma"
        )
        st.plotly_chart(fig_y, use_container_width=True)

    if behavioral_available:
        st.markdown("### Behavioral Prediction Performance (Pearson r)")

        render_normalized_kde_plot(
            r_z_arr, z_chan_names, split_res.get("block", []), "", use_bands=False
        )

        render_spectral_fvu_plot(
            split_res.get("Z", []),
            split_res.get("Zp", []),
            z_chan_names,
            "Behavioral Dynamics tracking (FVU)",
            sampling_freq=sampling_freq,
            use_bands=False,
        )

        render_distribution_comparison_plot(
            split_res.get("Z", []),
            split_res.get("Zp", []),
            z_chan_names,
            split_res.get("block", []),
            "",
            use_bands=False,
        )

        render_residual_acf_heatmap(
            split_res.get("Z", []),
            split_res.get("Zp", []),
            z_chan_names,
            "Behavioral Residuals Autocorrelation (ACF)",
            use_bands=False,
        )

        render_error_cdf_plot(
            split_res.get("Z", []),
            split_res.get("Zp", []),
            z_chan_names,
            "Behavioral Absolute Error CDF",
            use_bands=False,
        )

        render_residual_qq_plot(
            split_res.get("Z", []),
            split_res.get("Zp", []),
            z_chan_names,
            "Behavioral Residuals Normality (QQ Plot)",
            use_bands=False,
        )

        render_raincloud_plot(
            r_z_arr, z_chan_names, split_res.get("block", []), "", use_bands=False
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

        groups = []
        if use_bands:
            for name in active_channel_names:
                parts = str(name).split("_")
                if len(parts) >= 3:
                    found_band = parts[2].capitalize()
                elif len(parts) >= 2:
                    found_band = parts[1].capitalize()
                else:
                    found_band = "Unknown"
                groups.append(found_band)
        else:
            groups = active_channel_names

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

        df_plot = pd.DataFrame(data_list)

        import re

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

            fig.add_trace(
                go.Violin(
                    y=df_grp["Block"],
                    x=df_grp["Pearson r"],
                    name=grp,
                    legendgroup=grp,
                    showlegend=False,
                    line_color=color,
                    orientation="h",
                    marker=dict(size=3, opacity=0.7),
                    side="negative",
                    points="all",
                    jitter=0.5,
                    pointpos=-1.5,
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(width=0),
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
):
    if r_arr is None or len(channel_names) == 0:
        return

    n_trials, n_chans = r_arr.shape

    groups = []
    if use_bands:
        for name in channel_names:
            parts = str(name).split("_")
            if len(parts) >= 3:
                found_band = parts[2].capitalize()
            elif len(parts) >= 2:
                found_band = parts[1].capitalize()
            else:
                found_band = "Unknown"
            groups.append(found_band)
    else:
        groups = channel_names

    blocks_arr = (
        np.array(blocks) if (blocks and len(blocks) == n_trials) else np.zeros(n_trials)
    )
    unique_blks = np.unique(blocks_arr)

    z_data_list = []
    for blk in unique_blks:
        blk_mask = blocks_arr == blk
        blk_data = r_arr[blk_mask, :]

        mean = np.mean(blk_data)
        std = np.std(blk_data)

        if std > 1e-6:
            z_blk = (blk_data - mean) / std
        else:
            z_blk = blk_data - mean

        for t_sub_idx in range(z_blk.shape[0]):
            for c_idx in range(n_chans):
                z_data_list.append(
                    {
                        "Pearson r (Z-score)": z_blk[t_sub_idx, c_idx],
                        "Group": groups[c_idx],
                    }
                )

    df_z = pd.DataFrame(z_data_list)

    fig = px.violin(
        df_z,
        y="Pearson r (Z-score)",
        color="Group",
        box=True,
        points=False,
        title=title,
        template="plotly_white",
        labels={"Group": "Frequency Band" if use_bands else "Feature"},
    )

    fig.update_layout(height=500, hovermode="closest")

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Distribution of standardized performance across all blocks.")


def render_distribution_comparison_plot(
    data_true: list,
    data_pred: list,
    channel_names: list,
    blocks: list,
    title: str,
    use_bands: bool = False,
):
    if not data_true or not data_pred or len(channel_names) == 0:
        return

    n_trials = len(data_true)
    blocks_arr = (
        np.array(blocks) if (blocks and len(blocks) == n_trials) else np.zeros(n_trials)
    )
    unique_blks = np.unique(blocks_arr)

    n_chans = len(channel_names)
    groups_map = []
    if use_bands:
        for name in channel_names:
            parts = str(name).split("_")
            if len(parts) >= 3:
                found_band = parts[2].capitalize()
            elif len(parts) >= 2:
                found_band = parts[1].capitalize()
            else:
                found_band = "Unknown"
            groups_map.append(found_band)
    else:
        groups_map = channel_names

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

    for i, grp in enumerate(unique_groups):
        grp_indices = [idx for idx, g in enumerate(groups_map) if g == grp]

        pooled_true = []
        pooled_pred = []

        for blk in unique_blks:
            blk_indices = np.where(blocks_arr == blk)[0]

            blk_true_samples = []
            blk_pred_samples = []

            for t_idx in blk_indices:
                t_true = np.array(data_true[t_idx])
                t_pred = np.array(data_pred[t_idx])

                if t_true.ndim == 2:
                    blk_true_samples.append(t_true[:, grp_indices])
                    blk_pred_samples.append(t_pred[:, grp_indices])
                else:
                    blk_true_samples.append(t_true[grp_indices])
                    blk_pred_samples.append(t_pred[grp_indices])

            if not blk_true_samples:
                continue

            blk_true_flat = np.vstack(
                [np.atleast_2d(s) if s.ndim == 1 else s for s in blk_true_samples]
            )
            blk_pred_flat = np.vstack(
                [np.atleast_2d(s) if s.ndim == 1 else s for s in blk_pred_samples]
            )

            mean = np.mean(blk_true_flat)
            std = np.std(blk_true_flat)

            if std > 1e-6:
                pooled_true.append(((blk_true_flat - mean) / std).flatten())
                pooled_pred.append(((blk_pred_flat - mean) / std).flatten())
            else:
                pooled_true.append((blk_true_flat - mean).flatten())
                pooled_pred.append((blk_pred_flat - mean).flatten())

        if not pooled_true:
            continue

        final_true = np.concatenate(pooled_true)
        final_pred = np.concatenate(pooled_pred)

        # KS Test per panel
        ks_stat, p_val = ks_2samp(final_true, final_pred)

        # Subsample for speed
        max_pts = 5000
        if len(final_true) > max_pts:
            idx = np.random.choice(len(final_true), max_pts, replace=False)
            t_kde_data = final_true[idx]
            p_kde_data = final_pred[idx]
        else:
            t_kde_data = final_true
            p_kde_data = final_pred

        kde_true = gaussian_kde(t_kde_data)(x_range)
        kde_pred = gaussian_kde(p_kde_data)(x_range)

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

        # Add annotation for KS stat
        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref=f"x{i+1}" if i > 0 else "x",
            yref=f"y{i+1}" if i > 0 else "y",
            text=f"KS: {ks_stat:.2f}",
            showarrow=False,
            font=dict(size=10),
            align="right",
        )

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
    st.caption(
        "Standardized value distributions compared per group (pooled across blocks)."
    )


def render_spectral_fvu_plot(
    data_true: list,
    data_pred: list,
    channel_names: list,
    title: str,
    sampling_freq: float = 80.0,
    use_bands: bool = False,
):
    if not data_true or not data_pred or len(channel_names) == 0:
        return

    # 1. Filter and identify groups
    n_chans = len(channel_names)
    groups_map = []
    if use_bands:
        for name in channel_names:
            parts = str(name).split("_")
            if len(parts) >= 3:
                found_band = parts[2].capitalize()
            elif len(parts) >= 2:
                found_band = parts[1].capitalize()
            else:
                found_band = "Unknown"
            groups_map.append(found_band)
    else:
        groups_map = channel_names

    unique_groups = sorted(list(set(groups_map)))
    n_groups = len(unique_groups)

    # 2. Concatenate all trials for frequency analysis
    # trials are list of [Time, Channels]
    try:
        y_true_all = np.concatenate([np.atleast_2d(t) for t in data_true], axis=0)
        y_pred_all = np.concatenate([np.atleast_2d(t) for t in data_pred], axis=0)
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

        # FVU Ratio per channel
        # P_true is (Freq, ChansInGroup)
        ratios = P_resid / (P_true + 1e-10)

        # Stats across channels
        mean_ratio = np.mean(ratios, axis=1)
        sem_ratio = np.std(ratios, axis=1) / np.sqrt(max(1, ratios.shape[1]))

        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

        # Plot Mean Line
        fig.add_trace(
            go.Scatter(
                x=f,
                y=mean_ratio,
                line=dict(color="black", width=2),
                name=f"{grp} Mean",
                showlegend=False,
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

        # Baseline at 1.0 (Random performance)
        fig.add_shape(
            type="line",
            line=dict(color="red", width=1, dash="dash"),
            x0=0,
            x1=max(f),
            y0=1,
            y1=1,
            row=row,
            col=col,
            xref=f"x{i+1}" if i > 0 else "x",
            yref=f"y{i+1}" if i > 0 else "y",
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
    st.caption("Lower is better. 1.0 indicates model error matches signal power.")


def render_residual_acf_heatmap(
    data_true: list,
    data_pred: list,
    channel_names: list,
    title: str,
    use_bands: bool = False,
):
    if not data_true or not data_pred or not channel_names:
        return
    try:
        y_true = np.concatenate([np.atleast_2d(t) for t in data_true], axis=0)
        y_pred = np.concatenate([np.atleast_2d(t) for t in data_pred], axis=0)
        residuals = y_true - y_pred
        nlags = 20
        acf_matrix = np.zeros((len(channel_names), nlags + 1))
        for c in range(len(channel_names)):
            acf_matrix[c, :] = acf(residuals[:, c], nlags=nlags, fft=True)

        # Grouping for Y-axis
        if use_bands:
            y_labels = []
            for name in channel_names:
                parts = str(name).split("_")
                y_labels.append(
                    parts[2].capitalize()
                    if len(parts) >= 3
                    else parts[1].capitalize() if len(parts) >= 2 else "Unknown"
                )
            sort_idx = np.argsort(y_labels)
            acf_matrix = acf_matrix[sort_idx, :]
            y_labels = [y_labels[i] for i in sort_idx]
        else:
            y_labels = channel_names

        fig = go.Figure(
            data=go.Heatmap(
                z=acf_matrix[:, 1:],  # Skip lag 0
                x=list(range(1, nlags + 1)),
                y=y_labels,
                colorscale="RdBu",
                zmid=0,
                zmax=1,
                zmin=-1,
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Lag",
            yaxis_title="Channels",
            height=500,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Bright colors at lags > 0 indicate temporal structures the model failed to capture."
        )
    except Exception as e:
        st.error(f"ACF Error: {e}")


def render_error_cdf_plot(
    data_true: list,
    data_pred: list,
    channel_names: list,
    title: str,
    use_bands: bool = False,
):
    if not data_true or not data_pred or not channel_names:
        return
    try:
        y_true = np.concatenate([np.atleast_2d(t) for t in data_true], axis=0)
        y_pred = np.concatenate([np.atleast_2d(t) for t in data_pred], axis=0)
        abs_err = np.abs(y_true - y_pred)

        groups = []
        if use_bands:
            for name in channel_names:
                parts = str(name).split("_")
                groups.append(
                    parts[2].capitalize()
                    if len(parts) >= 3
                    else parts[1].capitalize() if len(parts) >= 2 else "Unknown"
                )
        else:
            groups = channel_names

        df_list = []
        for c in range(len(channel_names)):
            # Subsample for plot performance
            err_vals = abs_err[:, c]
            if len(err_vals) > 1000:
                err_vals = np.random.choice(err_vals, 1000, replace=False)
            for v in err_vals:
                df_list.append({"Abs Error": v, "Group": groups[c]})

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
):
    if not data_true or not data_pred or not channel_names:
        return
    try:
        y_true = np.concatenate([np.atleast_2d(t) for t in data_true], axis=0)
        y_pred = np.concatenate([np.atleast_2d(t) for t in data_pred], axis=0)
        resids = y_true - y_pred

        groups_map = []
        if use_bands:
            for name in channel_names:
                parts = str(name).split("_")
                groups_map.append(
                    parts[2].capitalize()
                    if len(parts) >= 3
                    else parts[1].capitalize() if len(parts) >= 2 else "Unknown"
                )
        else:
            groups_map = channel_names

        unique_groups = sorted(list(set(groups_map)))
        n_cols = 3
        n_rows = (len(unique_groups) + n_cols - 1) // n_cols
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=unique_groups)

        for i, grp in enumerate(unique_groups):
            grp_idx = [idx for idx, g in enumerate(groups_map) if g == grp]
            grp_resids = resids[:, grp_idx].flatten()
            # Standardize for QQ
            grp_resids = (grp_resids - np.mean(grp_resids)) / (
                np.std(grp_resids) + 1e-8
            )

            # Subsample
            if len(grp_resids) > 2000:
                grp_resids = np.random.choice(grp_resids, 2000, replace=False)

            osm, osr = probplot(grp_resids, dist="norm")[0]

            row, col = (i // n_cols) + 1, (i % n_cols) + 1
            fig.add_trace(
                go.Scatter(
                    x=osm,
                    y=osr,
                    mode="markers",
                    marker=dict(size=3, color="black", opacity=0.5),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=[-3, 3],
                    y=[-3, 3],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.update_layout(title=title, height=300 * n_rows, template="plotly_white")
        fig.update_xaxes(title_text="Theoretical Quantiles")
        fig.update_yaxes(title_text="Sample Quantiles")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "A straight line indicates Gaussian residuals. Deviations at ends reveal model failure on bursts."
        )
    except Exception as e:
        st.error(f"QQ Error: {e}")
