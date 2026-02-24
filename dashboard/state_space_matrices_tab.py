import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from utils.matrix_extraction import (
    load_model_matrices,
    analyze_eigenvalues,
    compare_matrices_across_conditions,
)


def render_matrix_heatmap(
    matrix: np.ndarray, title: str, key: str, colorscale: str = "RdBu_r"
):
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            colorscale=colorscale,
            zmid=0,
            colorbar=dict(title="Value"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Column Index",
        yaxis_title="Row Index",
        height=500,
        yaxis=dict(autorange="reversed"),
    )

    st.plotly_chart(fig, width="stretch", key=key)


def render_eigenvalue_plot(eig_analysis: Dict[str, Any], key: str):
    eigenvalues = eig_analysis["eigenvalues"]
    magnitudes = eig_analysis["magnitudes"]

    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)

    fig = go.Figure()

    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)
    fig.add_trace(
        go.Scatter(
            x=unit_circle_x,
            y=unit_circle_y,
            mode="lines",
            name="Unit Circle",
            line=dict(color="gray", dash="dash", width=2),
        )
    )

    colors = magnitudes
    fig.add_trace(
        go.Scatter(
            x=real_parts,
            y=imag_parts,
            mode="markers",
            name="Eigenvalues",
            marker=dict(
                size=12,
                color=colors,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Magnitude"),
                line=dict(color="black", width=1),
            ),
            text=[
                f"λ={ev:.3f}<br>|λ|={mag:.3f}"
                for ev, mag in zip(eigenvalues, magnitudes)
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Eigenvalues on Complex Plane",
        xaxis_title="Real Part",
        yaxis_title="Imaginary Part",
        height=500,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black"),
        yaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
            scaleanchor="x",
            scaleratio=1,
        ),
        showlegend=True,
    )

    st.plotly_chart(fig, width="stretch", key=f"eig_plot_{key}")


def render_eigenvalue_stats(eig_analysis: Dict[str, Any]):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Spectral Radius", f"{eig_analysis['spectral_radius']:.4f}")

    with col2:
        n_oscillatory = np.sum(eig_analysis["oscillatory_mask"])
        st.metric("Oscillatory Modes", n_oscillatory)

    with col3:
        if len(eig_analysis["oscillatory_freqs"]) > 0:
            dominant_freq = eig_analysis["oscillatory_freqs"][
                np.argmax(eig_analysis["oscillatory_mags"])
            ]
            st.metric("Dominant Frequency", f"{dominant_freq:.2f} Hz")
        else:
            st.metric("Dominant Frequency", "N/A")


def render_frequency_decay_table(eig_analysis: Dict[str, Any]):
    st.markdown("#### Oscillatory Modes")

    if len(eig_analysis["oscillatory_freqs"]) == 0:
        st.info("No oscillatory modes detected")
        return

    osc_indices = np.where(eig_analysis["oscillatory_mask"])[0]

    data = []
    for idx in osc_indices:
        data.append(
            {
                "Index": idx,
                "Frequency (Hz)": eig_analysis["frequencies_hz"][idx],
                "Magnitude": eig_analysis["magnitudes"][idx],
                "Decay Rate": eig_analysis["decay_rates"][idx],
                "Eigenvalue": f"{eig_analysis['eigenvalues'][idx]:.4f}",
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values("Magnitude", ascending=False)

    st.dataframe(df, width="stretch", hide_index=True)


def render_matrix_comparison(
    matrices_on: Dict[str, np.ndarray],
    matrices_off: Dict[str, np.ndarray],
    matrix_name: str,
    key: str,
):
    st.markdown(f"### {matrix_name} Matrix Comparison")

    if matrix_name not in matrices_on or matrix_name not in matrices_off:
        st.warning(f"{matrix_name} matrix not available in both conditions")
        return

    comparison = compare_matrices_across_conditions(
        matrices_on, matrices_off, matrix_name
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Frobenius Norm (ON)", f"{comparison['frobenius_norm_on']:.4f}")
    with col2:
        st.metric("Frobenius Norm (OFF)", f"{comparison['frobenius_norm_off']:.4f}")
    with col3:
        st.metric("Relative Difference", f"{comparison['relative_diff']:.4f}")

    st.metric("Max Absolute Difference", f"{comparison['max_abs_diff']:.4f}")

    tabs = st.tabs(["DBS ON", "DBS OFF", "Difference"])

    with tabs[0]:
        render_matrix_heatmap(
            matrices_on[matrix_name], f"{matrix_name} (DBS ON)", f"{key}_on"
        )

    with tabs[1]:
        render_matrix_heatmap(
            matrices_off[matrix_name], f"{matrix_name} (DBS OFF)", f"{key}_off"
        )

    with tabs[2]:
        render_matrix_heatmap(
            comparison["difference"],
            f"{matrix_name} Difference (ON - OFF)",
            f"{key}_diff",
            colorscale="RdBu_r",
        )


def state_space_matrices_tab(project_root, results_root=None):
    st.header("Learned Matrices Visualization")
    st.markdown(
        "Visualize the learned decoder matrices (Cy, Cz) from PSID and DPAD models."
    )

    RESULTS_ROOT = results_root if results_root else project_root / "results"

    from dashboard.subtabs import list_variants, list_run_timestamps

    variants = list_variants(RESULTS_ROOT)
    if len(variants) == 0:
        st.info("No result variants found under results/.")
        return

    variant = st.selectbox("Model variant", options=variants, key="matrix_variant")
    variant_dir = RESULTS_ROOT / variant
    runs = list_run_timestamps(variant_dir)

    if len(runs) == 0:
        st.info("No runs found for this variant yet. Train a model first.")
        return

    run_ts = st.selectbox("Run timestamp", options=runs, key="matrix_run")

    model_path = variant_dir / f"model_{run_ts}.pkl"

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return

    st.markdown("---")

    with st.spinner("Loading model and extracting matrices..."):
        try:
            matrices, model_type = load_model_matrices(model_path)
            st.success(
                f"Loaded {model_type.upper()} model with {len(matrices)} matrices"
            )
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    if len(matrices) == 0:
        st.warning(
            "No matrices found in model. The model may not have been trained or saved properly."
        )
        return

    st.markdown("### Available Matrices")
    matrix_info = []
    for name, matrix in matrices.items():
        matrix_info.append(
            {
                "Matrix": name,
                "Shape": f"{matrix.shape[0]} × {matrix.shape[1]}",
                "Size": matrix.size,
                "Dtype": str(matrix.dtype),
            }
        )

    df_info = pd.DataFrame(matrix_info)
    st.dataframe(df_info, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Matrix Visualizations")

    selected_matrix = st.selectbox(
        "Select matrix to visualize",
        options=list(matrices.keys()),
        key="selected_matrix_heatmap",
    )

    if selected_matrix:
        matrix = matrices[selected_matrix]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", matrix.shape[0])
        with col2:
            st.metric("Columns", matrix.shape[1])

        render_matrix_heatmap(
            matrix, f"{selected_matrix} Matrix", f"heatmap_{selected_matrix}"
        )

        with st.expander("Matrix Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{np.mean(matrix):.4f}")
            with col2:
                st.metric("Std", f"{np.std(matrix):.4f}")
            with col3:
                st.metric("Min", f"{np.min(matrix):.4f}")
            with col4:
                st.metric("Max", f"{np.max(matrix):.4f}")
