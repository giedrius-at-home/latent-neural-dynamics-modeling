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


def resolve_metric(res: Dict[str, Any], metric: str) -> float:
    """Resolve a metric value from a result dict, checking test_results first."""
    if "test_results" in res and metric in res["test_results"]:
        return res["test_results"][metric]
    if metric in res:
        return res[metric]
    return float("nan")


def reeval_against_history(res: Dict[str, Any], pred_res: Dict[str, Any]) -> bool:
    """
    Re-evaluate forecast results against historical predictions.
    Modifies `res` in-place. Returns True on success, False on failure.
    """
    if "y_pred" not in pred_res or "y_pred" not in res:
        return False
    history_preds = pred_res["y_pred"]
    y_pred = res["y_pred"]
    if len(history_preds) != len(y_pred):
        return False

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    y_true = history_preds
    res["accuracy"] = accuracy_score(y_true, y_pred)
    res["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    res["precision"] = precision_score(y_true, y_pred, zero_division=0)
    res["recall"] = recall_score(y_true, y_pred, zero_division=0)
    res["f1"] = f1_score(y_true, y_pred, zero_division=0)
    res["best_cv_score"] = res["balanced_accuracy"]
    res.pop("test_results", None)
    return True


def _apply_line_plot_layout(fig: go.Figure, title: str, xaxis_title: str) -> None:
    """Apply shared layout to accuracy-vs-parameter line plots."""
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color=PALETTE.cool_steel,
        annotation_text="Chance",
    )
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=PLOT_STYLE.title_size, family=PLOT_STYLE.font_family),
        ),
        xaxis=dict(
            title=dict(
                text=xaxis_title,
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
        ),
        yaxis=dict(
            title=dict(
                text="Balanced Accuracy",
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            range=[0.4, 1.0],
        ),
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family, color=PALETTE.ink_black),
        legend=dict(font=dict(size=PLOT_STYLE.tick_label_size)),
        margin=dict(l=60, r=60, t=80, b=60),
    )


def load_classification_results(
    results_dir: Path, mode: str = "forecast", eval_target: str = "dbs_stim"
) -> Dict[str, Dict[Tuple[float, float], Dict[str, Any]]]:
    """
    Loads all classification results, grouping them by FEATURE SOURCE.
    Returns: { feature_source: { (h, m): result_dict } }
    """
    all_results = {}
    hm_pattern = re.compile(r"^h([\d.]+)_m([\d.]+)$")

    # Traverse h/m directories
    for d in results_dir.iterdir():
        if not d.is_dir():
            continue

        hm_match = hm_pattern.match(d.name)
        if not hm_match:
            continue

        h_val, m_val = float(hm_match.group(1)), float(hm_match.group(2))

        # Find pkl files for the given mode (prediction, forecast, flipped)
        pattern_str = f"*_{mode}.pkl"
        pkl_files = list(d.glob(pattern_str))

        for pkl_file in pkl_files:
            filename = pkl_file.stem
            parts = filename.split("_")

            # Extract feature source: everything between LDA and mode/flipped
            suffixes = {"prediction", "forecast", "flipped", "epoch", "overlap"}
            feature_parts = []
            for p in parts[1:]:
                if any(p.startswith(s) for s in suffixes):
                    break
                feature_parts.append(p)

            feature_source = "_".join(feature_parts) if feature_parts else "Default"

            if feature_source not in all_results:
                all_results[feature_source] = {}

            try:
                with open(pkl_file, "rb") as f:
                    res = pickle.load(f)

                if eval_target == "history_label" and mode == "forecast":
                    pred_file = pkl_file.parent / pkl_file.name.replace(
                        "_forecast", "_prediction"
                    )
                    if pred_file.exists():
                        try:
                            with open(pred_file, "rb") as fp:
                                pred_res = pickle.load(fp)
                            reeval_against_history(res, pred_res)
                        except Exception as e:
                            st.warning(
                                f"Failed to load history prediction for {pkl_file.name}: {e}"
                            )

                all_results[feature_source][(h_val, m_val)] = res
            except Exception as e:
                st.warning(f"Failed to load {pkl_file}: {e}")

    return all_results


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

        z_matrix[m_idx, h_idx] = resolve_metric(res, metric)

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
                    font=dict(
                        size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                    ),
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
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
        ),
        yaxis=dict(
            title=dict(
                text="Forecast Horizon m (seconds)",
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
        ),
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family, color=PALETTE.ink_black),
        margin=dict(l=60, r=100, t=100, b=60),
    )

    return fig


def create_line_plot_by_history(
    all_results: Dict[str, Dict[Tuple[float, float], Dict[str, Any]]],
    metric: str = "balanced_accuracy",
) -> go.Figure:
    """
    Create line plot showing accuracy vs history length.
    Lines are added for each (feature_source, m) combination.
    """
    if not all_results:
        return go.Figure()

    fig = go.Figure()

    # Qualitative colors for different sources/windows
    colors = px.colors.qualitative.Plotly

    color_idx = 0
    for feature_source, results in all_results.items():
        h_values = sorted(set(hm[0] for hm in results.keys()))
        m_values = sorted(set(hm[1] for hm in results.keys()))

        for m in m_values:
            y_vals = []
            x_vals = []
            for h in h_values:
                if (h, m) in results:
                    y_vals.append(resolve_metric(results[(h, m)], metric))
                    x_vals.append(h)

            name = f"{feature_source} (m={m:.1f}s)"
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=colors[color_idx % len(colors)], width=2.5),
                    marker=dict(size=8),
                    hovertemplate=f"Feature: {feature_source}<br>m={m:.1f}s<br>h=%{{x:.1f}}s<br>Acc=%{{y:.3f}}<extra></extra>",
                )
            )
            color_idx += 1

    _apply_line_plot_layout(fig, "Accuracy vs History Length", "History h (seconds)")
    return fig


def create_line_plot_by_future(
    all_results: Dict[str, Dict[Tuple[float, float], Dict[str, Any]]],
    metric: str = "balanced_accuracy",
) -> go.Figure:
    if not all_results:
        return go.Figure()

    fig = go.Figure()
    colors = px.colors.qualitative.Safe

    color_idx = 0
    for feature_source, results in all_results.items():
        h_values = sorted(set(hm[0] for hm in results.keys()))
        m_values = sorted(set(hm[1] for hm in results.keys()))

        for h in h_values:
            y_vals = []
            x_vals = []
            for m in m_values:
                if (h, m) in results:
                    y_vals.append(resolve_metric(results[(h, m)], metric))
                    x_vals.append(m)

            name = f"{feature_source} (h={h:.1f}s)"
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=colors[color_idx % len(colors)], width=2.5),
                    marker=dict(size=8),
                    hovertemplate=f"Feature: {feature_source}<br>h={h:.1f}s<br>m=%{{x:.1f}}s<br>Acc=%{{y:.3f}}<extra></extra>",
                )
            )
            color_idx += 1

    _apply_line_plot_layout(
        fig, "Accuracy vs Forecast Horizon", "Forecast Horizon m (seconds)"
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
            marker=dict(
                size=15, color="white", line=dict(color=PALETTE.ink_black, width=2)
            ),
            text=["NOW"],
            textposition="top center",
            textfont=dict(
                size=14, family=PLOT_STYLE.font_family, color=PALETTE.ink_black
            ),
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
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
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
