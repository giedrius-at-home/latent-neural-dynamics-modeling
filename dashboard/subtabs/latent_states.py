import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any
from scipy.ndimage import gaussian_filter
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA as SklearnCCA
from umap import UMAP

from dashboard.subtabs.helpers import get_trial_time_axis
from utils.config import get_config
from utils.stats import (
    compute_power_spectrum,
    find_dominant_frequencies,
    spectral_correlation,
)

SAMPLING_FREQ = 60


def render_latent_states_plot(
    x_p: np.ndarray,
    t_abs: np.ndarray,
    t_offset: float,
    chunk_margin: Optional[float],
    duration: Optional[float],
    trial_idx: int,
):
    fig = go.Figure()
    t_x = (
        np.linspace(t_abs[0], t_abs[-1], x_p.shape[0])
        if len(t_abs) != x_p.shape[0]
        else t_abs
    )
    nx = x_p.shape[1] if x_p.ndim == 2 else 1
    x_min = float(np.nanmin(x_p))
    x_max = float(np.nanmax(x_p))

    for d in range(nx):
        series = x_p[:, d] if nx > 1 else x_p.squeeze()
        fig.add_trace(
            go.Scatter(
                x=t_x,
                y=series,
                name=f"X_p[{d}]",
                mode="lines",
            )
        )

    if duration is not None:
        event_start = (
            t_offset + float(chunk_margin) if chunk_margin is not None else t_x[0]
        )
        event_end = (
            t_offset
            + float(duration)
            - (float(chunk_margin) if chunk_margin is not None else 0.0)
        )
        fig.add_vrect(
            x0=event_start,
            x1=event_end,
            fillcolor="rgba(0, 100, 0, 0.1)",
            layer="below",
            line_width=0,
        )
        fig.add_vline(x=event_start, line_dash="dash", line_color="green")
        fig.add_vline(x=event_end, line_dash="dash", line_color="red")

    fig.update_layout(
        title=f"Latent states X_p — Trial {trial_idx}",
        xaxis_title="Time (s)",
        yaxis_title="Raw value",
        xaxis_range=[t_x[0], t_x[-1]],
        yaxis_range=[x_min, x_max],
    )
    st.plotly_chart(fig, use_container_width=True)


def render_auxiliary_predictions_plot(
    z_p: np.ndarray,
    t_abs: np.ndarray,
    t_offset: float,
    chunk_margin: Optional[float],
    duration: Optional[float],
    trial_idx: int,
):
    fig = go.Figure()
    t_z = (
        np.linspace(t_abs[0], t_abs[-1], z_p.shape[0])
        if len(t_abs) != z_p.shape[0]
        else t_abs
    )
    nz = z_p.shape[1] if z_p.ndim == 2 else 1
    z_min = float(np.nanmin(z_p))
    z_max = float(np.nanmax(z_p))

    for d in range(nz):
        series = z_p[:, d] if nz > 1 else z_p.squeeze()
        fig.add_trace(
            go.Scatter(
                x=t_z,
                y=series,
                name=f"Z_p[{d}]",
                mode="lines",
            )
        )

    if duration is not None:
        event_start = (
            t_offset + float(chunk_margin) if chunk_margin is not None else t_z[0]
        )
        event_end = (
            t_offset
            + float(duration)
            - (float(chunk_margin) if chunk_margin is not None else 0.0)
        )
        fig.add_vrect(
            x0=event_start,
            x1=event_end,
            fillcolor="rgba(0, 100, 0, 0.1)",
            layer="below",
            line_width=0,
        )
        fig.add_vline(x=event_start, line_dash="dash", line_color="green")
        fig.add_vline(x=event_end, line_dash="dash", line_color="red")

    fig.update_layout(
        title=f"Other predictions Z_p — Trial {trial_idx}",
        xaxis_title="Time (s)",
        yaxis_title="Value",
        xaxis_range=[t_z[0], t_z[-1]],
        yaxis_range=[z_min, z_max],
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tsne_umap_plot(
    x_p: np.ndarray, trial_idx: int, z_p: Optional[np.ndarray] = None
):

    if x_p.ndim != 2:
        st.warning("Latent states must be 2D for dimensionality reduction.")
        return

    n_samples, n_dims = x_p.shape

    if n_dims < 3:
        st.info(
            f"Latent space has only {n_dims} dimensions. Dimensionality reduction is most useful for higher-dimensional spaces."
        )
        return

    if z_p is not None and z_p.size > 0:
        if z_p.ndim == 2:
            color_values = z_p[:, 0]
            color_label = "Tracing Speed"
        else:
            color_values = z_p.squeeze()
            color_label = "Tracing Speed"

        if len(color_values) != n_samples:
            if len(color_values) > n_samples:
                color_values = color_values[:n_samples]
            else:
                color_values = np.arange(n_samples)
                color_label = "Time"
    else:
        color_values = np.arange(n_samples)
        color_label = "Time"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### t-SNE Projection")
        with st.spinner("Computing t-SNE..."):
            perplexity = min(30, n_samples - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            x_tsne = tsne.fit_transform(x_p)

        fig_tsne = go.Figure()
        fig_tsne.add_trace(
            go.Scatter(
                x=x_tsne[:, 0],
                y=x_tsne[:, 1],
                mode="markers",
                marker=dict(
                    size=4,
                    color=color_values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=color_label),
                ),
                text=[
                    f"t={i}, {color_label.lower()}={color_values[i]:.3f}"
                    for i in range(n_samples)
                ],
                hovertemplate="<b>Time step:</b> %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
            )
        )
        fig_tsne.update_layout(
            title=f"t-SNE of Latent States — Trial {trial_idx}",
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            height=400,
        )
        st.plotly_chart(fig_tsne, use_container_width=True)

    with col2:
        st.markdown("##### UMAP Projection")
        with st.spinner("Computing UMAP..."):
            n_neighbors = min(15, n_samples - 1)
            umap = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
            x_umap = umap.fit_transform(x_p)

        fig_umap = go.Figure()
        fig_umap.add_trace(
            go.Scatter(
                x=x_umap[:, 0],
                y=x_umap[:, 1],
                mode="markers",
                marker=dict(
                    size=4,
                    color=color_values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=color_label),
                ),
                text=[
                    f"t={i}, {color_label.lower()}={color_values[i]:.3f}"
                    for i in range(n_samples)
                ],
                hovertemplate="<b>Time step:</b> %{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
            )
        )
        fig_umap.update_layout(
            title=f"UMAP of Latent States — Trial {trial_idx}",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            height=400,
        )
        st.plotly_chart(fig_umap, use_container_width=True)


def render_cca_analysis(
    x_p: np.ndarray,
    z_data: np.ndarray,
    trial_idx: int,
    x_label: str = "Latent States (X)",
    z_label: str = "Behavioral Variable (Z)",
):

    if x_p.ndim != 2 or z_data.ndim != 2:
        st.warning("Both datasets must be 2D for CCA analysis.")
        return

    n_samples_x, n_features_x = x_p.shape
    n_samples_z, n_features_z = z_data.shape

    if n_samples_x != n_samples_z:
        st.warning(
            f"Sample size mismatch: {x_label} has {n_samples_x} samples, {z_label} has {n_samples_z} samples."
        )
        return

    n_samples = n_samples_x

    n_components = min(n_features_x, n_features_z, n_samples - 1)

    if n_components < 1:
        st.warning("Not enough components for CCA analysis.")
        return

    st.markdown(f"##### Canonical Correlation Analysis")
    st.markdown(
        f"Analyzing relationship between **{x_label}** ({n_features_x} dims) and **{z_label}** ({n_features_z} dims)"
    )

    with st.spinner("Computing CCA..."):
        cca = SklearnCCA(n_components=n_components)

        x_c, z_c = cca.fit_transform(x_p, z_data)

        canonical_corrs = np.array(
            [np.corrcoef(x_c[:, i], z_c[:, i])[0, 1] for i in range(n_components)]
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Canonical Correlations**")
        fig_corr = go.Figure()
        fig_corr.add_trace(
            go.Bar(
                x=list(range(1, n_components + 1)),
                y=canonical_corrs,
                marker_color="steelblue",
                text=[f"{corr:.3f}" for corr in canonical_corrs],
                textposition="outside",
            )
        )
        fig_corr.update_layout(
            title=f"Canonical Correlations — Trial {trial_idx}",
            xaxis_title="Canonical Component",
            yaxis_title="Correlation",
            yaxis_range=[0, 1.1],
            height=400,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.markdown("**Cumulative Variance Explained**")
        var_x = np.var(x_c, axis=0)
        var_z = np.var(z_c, axis=0)

        var_x_prop = var_x / np.sum(var_x) * 100
        var_z_prop = var_z / np.sum(var_z) * 100

        cum_var_x = np.cumsum(var_x_prop)
        cum_var_z = np.cumsum(var_z_prop)

        fig_var = go.Figure()
        fig_var.add_trace(
            go.Scatter(
                x=list(range(1, n_components + 1)),
                y=cum_var_x,
                mode="lines+markers",
                name=x_label,
                line=dict(color="blue", width=2),
                marker=dict(size=8),
            )
        )
        fig_var.add_trace(
            go.Scatter(
                x=list(range(1, n_components + 1)),
                y=cum_var_z,
                mode="lines+markers",
                name=z_label,
                line=dict(color="red", width=2),
                marker=dict(size=8),
            )
        )
        fig_var.update_layout(
            title=f"Cumulative Variance — Trial {trial_idx}",
            xaxis_title="Canonical Component",
            yaxis_title="Cumulative Variance (%)",
            yaxis_range=[0, 105],
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig_var, use_container_width=True)

    st.markdown("**Canonical Variate Relationships**")

    n_plot = min(2, n_components)

    if n_plot >= 1:
        cols = st.columns(n_plot)
        for i in range(n_plot):
            with cols[i]:
                fig_scatter = go.Figure()
                fig_scatter.add_trace(
                    go.Scatter(
                        x=x_c[:, i],
                        y=z_c[:, i],
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=np.arange(n_samples),
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Time"),
                        ),
                        text=[f"t={t}" for t in range(n_samples)],
                        hovertemplate=f"<b>Time:</b> %{{text}}<br>{x_label}: %{{x:.3f}}<br>{z_label}: %{{y:.3f}}<extra></extra>",
                    )
                )

                min_val = min(x_c[:, i].min(), z_c[:, i].min())
                max_val = max(x_c[:, i].max(), z_c[:, i].max())
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        line=dict(color="red", dash="dash", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                fig_scatter.update_layout(
                    title=f"Component {i+1} (r={canonical_corrs[i]:.3f})",
                    xaxis_title=f"{x_label} CC{i+1}",
                    yaxis_title=f"{z_label} CC{i+1}",
                    height=400,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("**CCA Loadings (Weights)**")
    st.markdown(
        "Shows how much each original variable contributes to each canonical component"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{x_label} Loadings**")
        x_loadings = np.corrcoef(x_p.T, x_c.T)[:n_features_x, n_features_x:]

        fig_load_x = go.Figure(
            data=go.Heatmap(
                z=x_loadings,
                x=[f"CC{i+1}" for i in range(n_components)],
                y=[f"Dim {i}" for i in range(n_features_x)],
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="Loading"),
            )
        )
        fig_load_x.update_layout(
            title=f"{x_label} Loadings",
            xaxis_title="Canonical Component",
            yaxis_title="Original Dimension",
            height=300,
        )
        st.plotly_chart(fig_load_x, use_container_width=True)

    with col2:
        st.markdown(f"**{z_label} Loadings**")
        z_loadings = np.corrcoef(z_data.T, z_c.T)[:n_features_z, n_features_z:]

        fig_load_z = go.Figure(
            data=go.Heatmap(
                z=z_loadings,
                x=[f"CC{i+1}" for i in range(n_components)],
                y=[f"Dim {i}" for i in range(n_features_z)],
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="Loading"),
            )
        )
        fig_load_z.update_layout(
            title=f"{z_label} Loadings",
            xaxis_title="Canonical Component",
            yaxis_title="Original Dimension",
            height=300,
        )
        st.plotly_chart(fig_load_z, use_container_width=True)

    st.markdown("**Summary Statistics**")
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Components", n_components)
    with stat_cols[1]:
        st.metric("Max Correlation", f"{canonical_corrs[0]:.3f}")
    with stat_cols[2]:
        st.metric("Mean Correlation", f"{np.mean(canonical_corrs):.3f}")
    with stat_cols[3]:
        strong_corr = np.sum(canonical_corrs > 0.5)
        st.metric("Strong Corr (r>0.5)", f"{strong_corr}/{n_components}")


def render_eigenvalue_analysis(x_p: np.ndarray, trial_idx: int):
    if x_p.ndim != 2:
        st.warning("Latent states must be 2D for eigenvalue analysis.")
        return

    n_samples, n_dims = x_p.shape

    x_centered = x_p - np.mean(x_p, axis=0)
    cov_matrix = np.cov(x_centered.T)

    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real

    total_var = np.sum(eigenvalues)
    if total_var == 0:
        explained_var = np.zeros_like(eigenvalues)
    else:
        explained_var = eigenvalues / total_var * 100
    cumulative_var = np.cumsum(explained_var)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Eigenvalue Spectrum")
        fig_eigen = go.Figure()
        fig_eigen.add_trace(
            go.Bar(
                x=list(range(1, len(eigenvalues) + 1)),
                y=eigenvalues,
                marker_color="steelblue",
                text=[f"{ev:.2f}" for ev in eigenvalues],
                textposition="outside",
            )
        )
        fig_eigen.update_layout(
            title=f"Eigenvalues of Latent Covariance — Trial {trial_idx}",
            xaxis_title="Component",
            yaxis_title="Eigenvalue",
            height=400,
        )
        st.plotly_chart(fig_eigen, use_container_width=True)

    with col2:
        st.markdown("##### Explained Variance")
        fig_var = go.Figure()
        fig_var.add_trace(
            go.Bar(
                x=list(range(1, len(explained_var) + 1)),
                y=explained_var,
                name="Individual",
                marker_color="lightblue",
            )
        )
        fig_var.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative_var) + 1)),
                y=cumulative_var,
                name="Cumulative",
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(size=8),
            )
        )
        fig_var.update_layout(
            title=f"Variance Explained — Trial {trial_idx}",
            xaxis_title="Component",
            yaxis_title="Variance Explained (%)",
            height=400,
            showlegend=True,
        )
        st.plotly_chart(fig_var, use_container_width=True)

    st.markdown("##### Statistics")
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Total Dimensions", n_dims)
    with stat_cols[1]:
        st.metric("Top Eigenvalue", f"{eigenvalues[0]:.2f}")
    with stat_cols[2]:
        n_95 = (
            np.argmax(cumulative_var >= 95) + 1
            if np.any(cumulative_var >= 95)
            else n_dims
        )
        st.metric("Dims for 95% Var", n_95)
    with stat_cols[3]:
        effective_dim = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues**2)
        st.metric("Effective Dimensionality", f"{effective_dim:.1f}")


def render_phase_space_analysis(
    split_res: Dict[str, Any],
    trial_idx: int,
    x_p: np.ndarray,
    z_p: np.ndarray = None,
):

    if x_p.ndim == 1:
        st.info("Phase space analysis requires at least 2 latent dimensions")
        return

    if x_p.shape[0] < x_p.shape[1]:
        x_p = x_p.T

    n_dims = x_p.shape[1]

    if n_dims < 2:
        st.info("Phase space analysis requires at least 2 latent dimensions")
        return

    col1, col2 = st.columns(2)
    with col1:
        dim_x = st.selectbox(
            "X-axis dimension",
            options=list(range(n_dims)),
            format_func=lambda x: f"Dimension {x+1}",
            key=f"phase_dim_x_{trial_idx}",
        )
    with col2:
        dim_y = st.selectbox(
            "Y-axis dimension",
            options=list(range(n_dims)),
            index=min(1, n_dims - 1),
            format_func=lambda x: f"Dimension {x+1}",
            key=f"phase_dim_y_{trial_idx}",
        )

    if dim_x == dim_y:
        st.warning("Please select different dimensions for each axis")
        return

    st.markdown("#### Single Trial Heatmap")
    _render_single_trial_heatmap(x_p, dim_x, dim_y, z_p, trial_idx)

    st.markdown("---")
    st.markdown("#### Trajectory with Arrows")
    render_trajectory(x_p, dim_x, dim_y, split_res, trial_idx, z_p=z_p)

    st.markdown("---")
    st.markdown("#### DBS ON/OFF Comparison")
    _render_dbs_comparison(split_res, dim_x, dim_y, trial_idx)


def _render_single_trial_heatmap(
    x_p: np.ndarray, dim_x: int, dim_y: int, z_p: np.ndarray, trial_idx: int
):

    x_data = x_p[:, dim_x]
    y_data = x_p[:, dim_y]

    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]

    if len(x_clean) < 10:
        st.warning("Not enough valid data points for heatmap")
        return

    n_bins = 50
    hist, x_edges, y_edges = np.histogram2d(x_clean, y_clean, bins=n_bins)
    hist_smooth = gaussian_filter(hist.T, sigma=1.5)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=x_edges[:-1],
            y=y_edges[:-1],
            z=hist_smooth,
            colorscale="Viridis",
            colorbar=dict(title="Density"),
            hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Density: %{z:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Phase Space Heatmap: Dimension {dim_x+1} vs Dimension {dim_y+1}",
        xaxis_title=f"Latent Dimension {dim_x+1}",
        yaxis_title=f"Latent Dimension {dim_y+1}",
        showlegend=False,
        hovermode="closest",
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_dbs_comparison(
    split_res: Dict[str, Any], dim_x: int, dim_y: int, trial_idx: int
):

    Xp_list = split_res.get("Xp", [])
    stim_list = split_res.get("stim", [])

    if not Xp_list or not stim_list:
        st.warning("DBS stimulation data not available")
        return

    dbs_on_trials = []
    dbs_off_trials = []

    for idx, stim in enumerate(stim_list):
        if idx >= len(Xp_list):
            continue
        if stim == "on":
            dbs_on_trials.append(idx)
        elif stim == "off":
            dbs_off_trials.append(idx)

    if not dbs_on_trials and not dbs_off_trials:
        st.warning("No DBS trials available for comparison")
        return

    if not dbs_on_trials:
        st.info(f"Only DBS OFF trials available ({len(dbs_off_trials)} trials)")
    elif not dbs_off_trials:
        st.info(f"Only DBS ON trials available ({len(dbs_on_trials)} trials)")

    def aggregate_phase_space(trial_indices):
        all_x = []
        all_y = []
        for idx in trial_indices:
            x_trial = np.array(Xp_list[idx])
            if x_trial.shape[0] < x_trial.shape[1]:
                x_trial = x_trial.T
            if x_trial.shape[1] <= max(dim_x, dim_y):
                continue
            x_data = x_trial[:, dim_x]
            y_data = x_trial[:, dim_y]
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
            all_x.extend(x_data[valid_mask])
            all_y.extend(y_data[valid_mask])
        return np.array(all_x), np.array(all_y)

    x_on, y_on = (
        aggregate_phase_space(dbs_on_trials)
        if dbs_on_trials
        else (np.array([]), np.array([]))
    )
    x_off, y_off = (
        aggregate_phase_space(dbs_off_trials)
        if dbs_off_trials
        else (np.array([]), np.array([]))
    )

    has_on = len(x_on) >= 10
    has_off = len(x_off) >= 10

    if not has_on and not has_off:
        st.warning("Not enough data points for visualization")
        return

    n_bins = 50

    all_x = []
    all_y = []
    if has_on:
        all_x.extend(x_on)
        all_y.extend(y_on)
    if has_off:
        all_x.extend(x_off)
        all_y.extend(y_off)

    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)

    x_bins = np.linspace(x_min, x_max, n_bins)
    y_bins = np.linspace(y_min, y_max, n_bins)

    hist_on_smooth = None
    hist_off_smooth = None

    if has_on:
        hist_on, _, _ = np.histogram2d(x_on, y_on, bins=[x_bins, y_bins])
        hist_on_smooth = gaussian_filter(hist_on.T, sigma=1.5)

    if has_off:
        hist_off, _, _ = np.histogram2d(x_off, y_off, bins=[x_bins, y_bins])
        hist_off_smooth = gaussian_filter(hist_off.T, sigma=1.5)

    if has_on and has_off:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"DBS ON ({len(dbs_on_trials)} trials)",
                f"DBS OFF ({len(dbs_off_trials)} trials)",
            ),
            horizontal_spacing=0.12,
        )

        fig.add_trace(
            go.Heatmap(
                x=x_bins[:-1],
                y=y_bins[:-1],
                z=hist_on_smooth,
                colorscale="Viridis",
                colorbar=dict(title="Density", x=0.45),
                hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Density: %{z:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=x_bins[:-1],
                y=y_bins[:-1],
                z=hist_off_smooth,
                colorscale="Viridis",
                colorbar=dict(title="Density", x=1.02),
                hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Density: %{z:.1f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text=f"Latent Dimension {dim_x+1}", row=1, col=1)
        fig.update_xaxes(title_text=f"Latent Dimension {dim_x+1}", row=1, col=2)
        fig.update_yaxes(title_text=f"Latent Dimension {dim_y+1}", row=1, col=1)
        fig.update_yaxes(title_text=f"Latent Dimension {dim_y+1}", row=1, col=2)

        fig.update_layout(
            title_text=f"DBS ON/OFF Phase Space Comparison: Dim {dim_x+1} vs Dim {dim_y+1}",
            height=600,
        )
    else:
        condition = "on" if has_on else "off"
        n_trials = len(dbs_on_trials) if has_on else len(dbs_off_trials)
        hist_smooth = hist_on_smooth if has_on else hist_off_smooth

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                x=x_bins[:-1],
                y=y_bins[:-1],
                z=hist_smooth,
                colorscale="Viridis",
                colorbar=dict(title="Density"),
                hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Density: %{z:.1f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"DBS {condition} Phase Space: Dim {dim_x+1} vs Dim {dim_y+1} ({n_trials} trials)",
            xaxis_title=f"Latent Dimension {dim_x+1}",
            yaxis_title=f"Latent Dimension {dim_y+1}",
            height=600,
        )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Distribution Statistics")

    if has_on and has_off:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**DBS ON**")
            st.write(f"Mean X: {x_on.mean():.4f} ± {x_on.std():.4f}")
            st.write(f"Mean Y: {y_on.mean():.4f} ± {y_on.std():.4f}")
        with col2:
            st.markdown("**DBS OFF**")
            st.write(f"Mean X: {x_off.mean():.4f} ± {x_off.std():.4f}")
            st.write(f"Mean Y: {y_off.mean():.4f} ± {y_off.std():.4f}")
    elif has_on:
        st.markdown("**DBS ON**")
        st.write(f"Mean X: {x_on.mean():.4f} ± {x_on.std():.4f}")
        st.write(f"Mean Y: {y_on.mean():.4f} ± {y_on.std():.4f}")
    else:
        st.markdown("**DBS OFF**")
        st.write(f"Mean X: {x_off.mean():.4f} ± {x_off.std():.4f}")
        st.write(f"Mean Y: {y_off.mean():.4f} ± {y_off.std():.4f}")


def render_latent_frequency_analysis(
    x_p: np.ndarray,
    y_true: np.ndarray,
    trial_idx: int,
    sampling_freq: float = SAMPLING_FREQ,
):

    st.markdown("### Frequency Analysis of Latent Dynamics")
    st.markdown("Compare frequency content of latent states vs neural signals")

    if x_p.ndim == 1:
        x_p = x_p.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    if x_p.shape[0] < x_p.shape[1]:
        x_p = x_p.T
    if y_true.shape[0] < y_true.shape[1]:
        y_true = y_true.T

    n_latent_dims = x_p.shape[1]
    n_neural_chans = y_true.shape[1]

    col1, col2 = st.columns(2)
    with col1:
        latent_dim = st.selectbox(
            "Latent dimension",
            options=list(range(n_latent_dims)),
            format_func=lambda x: f"Dimension {x+1}",
            key=f"latent_freq_dim_{trial_idx}",
        )
    with col2:
        neural_chan = st.selectbox(
            "Neural channel",
            options=list(range(n_neural_chans)),
            format_func=lambda x: f"Channel {x+1}",
            key=f"latent_freq_chan_{trial_idx}",
        )

    x_signal = x_p[:, latent_dim]
    y_signal = y_true[:, neural_chan]

    min_len = min(len(x_signal), len(y_signal))
    x_signal = x_signal[:min_len]
    y_signal = y_signal[:min_len]

    freqs_x, psd_x = compute_power_spectrum(x_signal, sampling_freq)
    freqs_y, psd_y = compute_power_spectrum(y_signal, sampling_freq)

    dom_freqs_x, dom_powers_x = find_dominant_frequencies(freqs_x, psd_x, n_peaks=5)
    dom_freqs_y, dom_powers_y = find_dominant_frequencies(freqs_y, psd_y, n_peaks=5)

    spec_corr = spectral_correlation(freqs_x, psd_x, psd_y)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=freqs_x,
            y=psd_x,
            mode="lines",
            name=f"Latent Dim {latent_dim+1}",
            line=dict(color="purple", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=freqs_y,
            y=psd_y,
            mode="lines",
            name=f"Neural Chan {neural_chan+1}",
            line=dict(color="orange", width=2, dash="dash"),
        )
    )

    if len(dom_freqs_x) > 0:
        fig.add_trace(
            go.Scatter(
                x=dom_freqs_x,
                y=dom_powers_x,
                mode="markers",
                name="Latent Peaks",
                marker=dict(size=10, color="purple", symbol="x"),
            )
        )

    if len(dom_freqs_y) > 0:
        fig.add_trace(
            go.Scatter(
                x=dom_freqs_y,
                y=dom_powers_y,
                mode="markers",
                name="Neural Peaks",
                marker=dict(size=10, color="orange", symbol="circle"),
            )
        )

    fig.update_layout(
        title=f"Power Spectrum: Latent Dim {latent_dim+1} vs Neural Chan {neural_chan+1}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density",
        yaxis_type="log",
        showlegend=True,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### Dominant Frequencies (Latent Dim {latent_dim+1})")
        if len(dom_freqs_x) > 0:
            for i, (freq, power) in enumerate(zip(dom_freqs_x, dom_powers_x)):
                st.metric(f"Peak {i+1}", f"{freq:.2f} Hz", f"Power: {power:.2e}")
        else:
            st.info("No dominant peaks detected")

    with col2:
        st.markdown(f"#### Dominant Frequencies (Neural Chan {neural_chan+1})")
        if len(dom_freqs_y) > 0:
            for i, (freq, power) in enumerate(zip(dom_freqs_y, dom_powers_y)):
                st.metric(f"Peak {i+1}", f"{freq:.2f} Hz", f"Power: {power:.2e}")
        else:
            st.info("No dominant peaks detected")

    st.markdown("#### Spectral Correlation")
    st.metric(
        "Correlation between spectra",
        f"{spec_corr:.4f}",
    )


def render_trajectory(
    x_p: np.ndarray,
    dim_x: int,
    dim_y: int,
    split_res: Dict[str, Any],
    trial_idx: int,
    z_p: Optional[np.ndarray] = None,
    step: int = 10,
):
    x, y = x_p[:, dim_x], x_p[:, dim_y]
    dx, dy = np.diff(x), np.diff(y)

    t_abs = (
        split_res.get("time_abs", [None])[trial_idx]
        if trial_idx < len(split_res.get("time_abs", []))
        else None
    )

    if z_p is not None and z_p.ndim == 2 and z_p.shape[1] > 1:
        z_channel_options = split_res.get(
            "output_channels", [f"z_ch{i}" for i in range(z_p.shape[1])]
        )
        selected_z_ch = st.selectbox(
            "Color trajectory by:",
            options=range(z_p.shape[1]),
            format_func=lambda i: (
                z_channel_options[i] if i < len(z_channel_options) else f"Z[{i}]"
            ),
            key=f"traj_color_z_{trial_idx}",
        )
        color_values = z_p[:, selected_z_ch]
        color_label = z_channel_options[selected_z_ch]
    else:
        if z_p is not None and z_p.size > 0:
            output_chan_names = split_res.get("output_channels", [])
            color_values = z_p[:, 0] if z_p.ndim == 2 else z_p
            color_label = output_chan_names[0] if output_chan_names else "Z_p"
        elif t_abs is not None:
            color_values = t_abs
            color_label = "Time (s)"
        else:
            color_values = np.arange(len(x))
            color_label = "Sample Index"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="steelblue", width=2),
            showlegend=False,
            hovertemplate="x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>",
        )
    )

    arrow_step = max(step, len(x) // 2)
    arrow_indices = np.arange(0, len(dx), arrow_step)

    for i in arrow_indices:
        if i + 1 < len(x):
            fig.add_annotation(
                x=x[i + 1],
                y=y[i + 1],
                ax=x[i],
                ay=y[i],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="rgba(70, 130, 180, 0.7)",
            )
    fig.add_trace(
        go.Scatter(
            x=[x[0]],
            y=[y[0]],
            mode="markers",
            marker=dict(
                size=15,
                color="green",
                symbol="circle",
                line=dict(width=2, color="darkgreen"),
            ),
            name="Start",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[x[-1]],
            y=[y[-1]],
            mode="markers",
            marker=dict(
                size=15,
                color="red",
                symbol="square",
                line=dict(width=2, color="darkred"),
            ),
            name="End",
        )
    )

    fig.update_layout(
        title=f"Trajectory: Dim {dim_x+1} vs Dim {dim_y+1} — Trial {trial_idx}",
        xaxis_title=f"Latent Dimension {dim_x+1}",
        yaxis_title=f"Latent Dimension {dim_y+1}",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_latent_states_tab(split_res: Dict[str, Any], trial_idx: int):
    Xp = split_res["Xp"]
    Zp = split_res["Zp"]

    x_p = np.array(Xp[trial_idx])
    z_p = None if Zp[trial_idx] is None else np.array(Zp[trial_idx])

    Z_true = split_res.get("Z", [])
    z_true = None
    if Z_true and len(Z_true) > trial_idx and Z_true[trial_idx] is not None:
        z_true = np.array(Z_true[trial_idx])

    offsets = split_res.get("offset", [])
    t_offset = (
        float(offsets[trial_idx])
        if offsets and len(offsets) > trial_idx and offsets[trial_idx] is not None
        else 0.0
    )

    Y_true = split_res["Y"]
    y_t = np.array(Y_true[trial_idx])
    n_samples = y_t.shape[0]
    t_abs = get_trial_time_axis(split_res, trial_idx, n_samples, t_offset)

    cm_list = split_res.get("chunk_margin", [])
    md_list = split_res.get("margined_duration", [])
    cm = cm_list[trial_idx] if cm_list else None
    dur = md_list[trial_idx] if md_list else None

    if x_p is not None:
        st.markdown("### Time Series")
        st.markdown("Latent state trajectories over time")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Latent States (X_p)")
            render_latent_states_plot(x_p, t_abs, t_offset, cm, dur, trial_idx)

        with col2:
            if z_p is not None:
                st.markdown("#### Other Predictions (Z_p)")
                render_auxiliary_predictions_plot(
                    z_p, t_abs, t_offset, cm, dur, trial_idx
                )

        st.markdown("---")
        st.markdown("### Dimensionality Reduction")
        st.markdown("Low-dimensional projections of latent space")
        render_tsne_umap_plot(x_p, trial_idx, z_p=z_true)

        st.markdown("---")
        st.markdown("### Eigenvalue Analysis")
        render_eigenvalue_analysis(x_p, trial_idx)

        st.markdown("---")
        st.markdown("### Phase Space Analysis")
        st.markdown("Comprehensive phase space visualization with multiple views")
        render_phase_space_analysis(split_res, trial_idx, x_p, z_p=z_true)

        st.markdown("---")
        cfg_path = st.session_state.get("config_path")
        cfg = get_config(str(cfg_path))
        sampling_freq = cfg.data.sampling_frequency

        render_latent_frequency_analysis(
            x_p,
            y_t,
            trial_idx,
            sampling_freq,
        )

        st.markdown("---")
        st.markdown("### Canonical Correlation Analysis")
        st.markdown("Analyze relationships between latent states and other variables")

        cca_options = []
        if z_true is not None and z_true.ndim == 2:
            cca_options.append("Latent States (X) vs Behavioral Variables (Z)")
        if y_t is not None and y_t.ndim == 2:
            cca_options.append("Latent States (X) vs Neural Signals (Y)")

        if len(cca_options) > 0:
            cca_choice = st.selectbox(
                "Select CCA analysis",
                options=cca_options,
                key=f"cca_choice_{trial_idx}",
            )

            if cca_choice == "Latent States (X) vs Behavioral Variables (Z)":
                render_cca_analysis(
                    x_p,
                    z_true,
                    trial_idx,
                    x_label="Latent States (X)",
                    z_label="Behavioral Variables (Z)",
                )
            elif cca_choice == "Latent States (X) vs Neural Signals (Y)":
                render_cca_analysis(
                    x_p,
                    y_t,
                    trial_idx,
                    x_label="Latent States (X)",
                    z_label="Neural Signals (Y)",
                )
        else:
            st.info(
                "CCA analysis requires at least one additional variable (Z or Y) with multiple dimensions."
            )

    else:
        st.info("No latent states available for this trial.")
