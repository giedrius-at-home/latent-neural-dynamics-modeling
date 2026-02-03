
import streamlit as st
import numpy as np
import pickle
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dashboard.backbone import (
    PALETTE,
    PLOT_COLOR,
    PLOT_STYLE,
    update_fig_title,
)


def load_classification_results(
    results_dir: Path,
) -> Dict[Tuple[float, float], Dict[str, Any]]:
    results = {}
    pattern = re.compile(r"^h([\d.]+)_m([\d.]+)$")
    
    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        match = pattern.match(d.name)
        if not match:
            continue
            
        h_val = float(match.group(1))
        m_val = float(match.group(2))
        
        # Look for LDA results (primary classifier)
        pkl_files = list(d.glob("LDA_*.pkl"))
        if not pkl_files:
            continue
            
        try:
            with open(pkl_files[0], "rb") as f:
                result = pickle.load(f)
            results[(h_val, m_val)] = result
        except Exception as e:
            st.warning(f"Failed to load {pkl_files[0]}: {e}")
            
    return results


def create_heatmap_figure(
    results: Dict[Tuple[float, float], Dict[str, Any]],
    metric: str = "balanced_accuracy",
    title: str = "Classification Performance",
) -> go.Figure:
    """
    Create a heatmap showing classification performance across h and m values.
    """
    if not results:
        return go.Figure()
    
    # Extract unique h and m values
    h_values = sorted(set(hm[0] for hm in results.keys()))
    m_values = sorted(set(hm[1] for hm in results.keys()))
    
    # Build the z-matrix
    z_matrix = np.full((len(m_values), len(h_values)), np.nan)
    
    for (h, m), res in results.items():
        h_idx = h_values.index(h)
        m_idx = m_values.index(m)
        
        # Get the metric value (prefer test results if available)
        if "test_results" in res and metric in res["test_results"]:
            z_matrix[m_idx, h_idx] = res["test_results"][metric]
        elif metric in res:
            z_matrix[m_idx, h_idx] = res[metric]
        elif metric == "balanced_accuracy" and "best_cv_score" in res:
            z_matrix[m_idx, h_idx] = res["best_cv_score"]
    
    # Find best cell
    best_idx = np.nanargmax(z_matrix)
    best_m_idx, best_h_idx = np.unravel_index(best_idx, z_matrix.shape)
    best_h = h_values[best_h_idx]
    best_m = m_values[best_m_idx]
    best_val = z_matrix[best_m_idx, best_h_idx]
    
    # Create heatmap
    fig = go.Figure()
    
    fig.add_trace(
        go.Heatmap(
            z=z_matrix,
            x=[f"{h:.1f}" for h in h_values],
            y=[f"{m:.1f}" for m in m_values],
            colorscale=[
                [0.0, PALETTE.twilight_indigo],
                [0.5, PALETTE.vintage_grape],
                [1.0, PALETTE.strawberry_red],
            ],
            colorbar=dict(
                title=dict(
                    text="Balanced<br>Accuracy",
                    font=dict(size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family),
                ),
                tickfont=dict(size=PLOT_STYLE.tick_label_size),
            ),
            hovertemplate="h=%{x}s, m=%{y}s<br>Accuracy=%{z:.3f}<extra></extra>",
            zmin=0.5,  # Chance level
            zmax=1.0,
        )
    )
    
    # Add annotation for best cell
    fig.add_annotation(
        x=f"{best_h:.1f}",
        y=f"{best_m:.1f}",
        text="★",
        showarrow=False,
        font=dict(size=20, color="white"),
    )
    
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>Best: h={best_h:.1f}s, m={best_m:.1f}s (acc={best_val:.3f})</sup>",
            x=0.5,
            xanchor="center",
            font=dict(size=PLOT_STYLE.title_size, family=PLOT_STYLE.font_family),
        ),
        xaxis=dict(
            title=dict(
                text="History h (seconds)",
                font=dict(size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
        ),
        yaxis=dict(
            title=dict(
                text="Forecast Horizon m (seconds)",
                font=dict(size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
        ),
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family, color=PALETTE.ink_black),
        margin=dict(l=60, r=100, t=100, b=60),
    )
    
    return fig


def create_line_plot_by_history(
    results: Dict[Tuple[float, float], Dict[str, Any]],
    metric: str = "balanced_accuracy",
) -> go.Figure:
    """
    Create line plot showing accuracy vs history length, with separate lines for each m.
    """
    if not results:
        return go.Figure()
    
    h_values = sorted(set(hm[0] for hm in results.keys()))
    m_values = sorted(set(hm[1] for hm in results.keys()))
    
    # Color scale for different m values
    colors = px.colors.qualitative.Set2[:len(m_values)]
    
    fig = go.Figure()
    
    for i, m in enumerate(m_values):
        y_vals = []
        x_vals = []
        for h in h_values:
            if (h, m) in results:
                res = results[(h, m)]
                if "test_results" in res and metric in res["test_results"]:
                    val = res["test_results"][metric]
                elif metric in res:
                    val = res[metric]
                elif metric == "balanced_accuracy" and "best_cv_score" in res:
                    val = res["best_cv_score"]
                else:
                    val = np.nan
                y_vals.append(val)
                x_vals.append(h)
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=f"m = {m:.1f}s",
                line=dict(color=colors[i % len(colors)], width=2.5),
                marker=dict(size=8),
                hovertemplate=f"m={m:.1f}s<br>h=%{{x:.1f}}s<br>Accuracy=%{{y:.3f}}<extra></extra>",
            )
        )
    
    # Add chance level line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color=PALETTE.cool_steel,
        annotation_text="Chance",
        annotation_position="bottom right",
    )
    
    fig.update_layout(
        title=dict(
            text="Classification Accuracy vs History Length",
            x=0.5,
            xanchor="center",
            font=dict(size=PLOT_STYLE.title_size, family=PLOT_STYLE.font_family),
        ),
        xaxis=dict(
            title=dict(
                text="History h (seconds)",
                font=dict(size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
        ),
        yaxis=dict(
            title=dict(
                text="Balanced Accuracy",
                font=dict(size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            range=[0.4, 1.0],
        ),
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family, color=PALETTE.ink_black),
        legend=dict(
            title=dict(text="Forecast Horizon"),
            font=dict(size=PLOT_STYLE.tick_label_size),
        ),
        margin=dict(l=60, r=60, t=80, b=60),
    )
    
    return fig


def create_timeline_visualization(
    best_h: float,
    best_m: float,
    best_accuracy: float,
) -> go.Figure:
    """
    Create a timeline showing the optimal history and forecast windows relative to "now".
    """
    fig = go.Figure()
    
    # Timeline axis
    t_min = -best_h - 0.5
    t_max = best_m + 0.5
    
    # History window (past)
    fig.add_trace(
        go.Scatter(
            x=[-best_h, 0],
            y=[0, 0],
            mode="lines",
            line=dict(color=PALETTE.vintage_grape, width=20),
            name=f"History ({best_h:.1f}s)",
            hoverinfo="name",
        )
    )
    
    # Forecast window (future)
    fig.add_trace(
        go.Scatter(
            x=[0, best_m],
            y=[0, 0],
            mode="lines",
            line=dict(color=PALETTE.strawberry_red, width=20),
            name=f"Forecast ({best_m:.1f}s)",
            hoverinfo="name",
        )
    )
    
    # "Now" marker
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers+text",
            marker=dict(size=15, color="white", line=dict(color=PALETTE.ink_black, width=2)),
            text=["NOW"],
            textposition="top center",
            textfont=dict(size=14, family=PLOT_STYLE.font_family, color=PALETTE.ink_black),
            name="Decision Point",
            hoverinfo="name",
        )
    )
    
    # Add arrows and labels
    fig.add_annotation(
        x=-best_h / 2,
        y=-0.15,
        text=f"← {best_h:.1f}s history",
        showarrow=False,
        font=dict(size=12, family=PLOT_STYLE.font_family, color=PALETTE.vintage_grape),
    )
    
    fig.add_annotation(
        x=best_m / 2,
        y=-0.15,
        text=f"{best_m:.1f}s forecast →",
        showarrow=False,
        font=dict(size=12, family=PLOT_STYLE.font_family, color=PALETTE.strawberry_red),
    )
    
    fig.update_layout(
        title=dict(
            text=f"Optimal Classification Window (Accuracy: {best_accuracy:.1%})",
            x=0.5,
            xanchor="center",
            font=dict(size=PLOT_STYLE.title_size, family=PLOT_STYLE.font_family),
        ),
        xaxis=dict(
            title=dict(
                text="Time (seconds)",
                font=dict(size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            range=[t_min, t_max],
            zeroline=True,
            zerolinecolor=PALETTE.cool_steel,
            zerolinewidth=1,
        ),
        yaxis=dict(
            visible=False,
            range=[-0.5, 0.5],
        ),
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family, color=PALETTE.ink_black),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        height=250,
        margin=dict(l=60, r=60, t=100, b=60),
    )
    
    return fig


def create_summary_table(
    results: Dict[Tuple[float, float], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create a summary table of all h/m results."""
    rows = []
    for (h, m), res in sorted(results.items()):
        row = {
            "h (s)": h,
            "m (s)": m,
            "CV Score": res.get("best_cv_score", np.nan),
            "Balanced Acc": res.get("balanced_accuracy", np.nan),
        }
        
        if "test_results" in res:
            row["Test Acc"] = res["test_results"].get("balanced_accuracy", np.nan)
        
        if "permutation_test" in res:
            row["p-value"] = res["permutation_test"].get("pvalue", np.nan)
            
        rows.append(row)
    
    return rows


def render_classification_tab(variant_dir: Path):

    st.markdown("## Classification Results on Different History and Forecast Windows")
    
    # Load results
    results = load_classification_results(variant_dir)
    
    if not results:
        st.warning(f"No classification results found in {variant_dir}")
        st.info("Run classification with h and m parameters to generate results.")
        return
    
    # Find best configuration
    best_hm = max(results.keys(), key=lambda hm: results[hm].get("best_cv_score", 0))
    best_h, best_m = best_hm
    best_acc = results[best_hm].get("best_cv_score", 0)
    
    # Metric selector
    col1, col2 = st.columns([1, 3])
    with col1:
        metric = st.selectbox(
            "Metric",
            ["best_cv_score", "balanced_accuracy", "accuracy", "f1"],
            format_func=lambda x: {
                "best_cv_score": "CV Balanced Accuracy",
                "balanced_accuracy": "Train Balanced Accuracy",
                "accuracy": "Train Accuracy",
                "f1": "F1 Score",
            }.get(x, x),
        )
    
    st.markdown("### Heatmap")
    fig_heatmap = create_heatmap_figure(results, metric=metric)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("### Performance by History Length")
    fig_lines = create_line_plot_by_history(results, metric=metric)
    st.plotly_chart(fig_lines, use_container_width=True)
    
    st.markdown("### Best Configuration Timeline")
    fig_timeline = create_timeline_visualization(best_h, best_m, best_acc)
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    st.info(f"""
    **Optimal Configuration:**
    - Use **{best_h:.1f} seconds** of history before the decision point
    - Predict **{best_m:.1f} seconds** into the future
    - Achieves **{best_acc:.1%}** balanced accuracy
    """)
    
    # st.markdown("### Detailed Results Table")
    # import pandas as pd
    # table_data = create_summary_table(results)
    # df = pd.DataFrame(table_data)
    
    # st.dataframe(
    #     df.style.format({
    #         "CV Score": "{:.3f}",
    #         "Balanced Acc": "{:.3f}",
    #         "Test Acc": "{:.3f}",
    #         "p-value": "{:.4f}",
    #     }).background_gradient(subset=["CV Score"], cmap="RdYlGn", vmin=0.5, vmax=1.0),
    #     use_container_width=True,
    # )
