import streamlit as st
import numpy as np
import pickle
import re
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from dashboard.backbone import (
    PALETTE,
    PLOT_COLOR,
    PLOT_STYLE,
    update_fig_title,
)


def resolve_metric(
    res: Dict[str, Any], metric: str, use_training_only: bool = False
) -> float:
    """
    Resolve a metric value from results dictionary.

    Args:
        res: Results dictionary
        metric: Metric name to resolve
        use_training_only: If True, skip test_results and only use top-level metrics

    Returns:
        Metric value or NaN if not found
    """
    if metric == "best_cv_score":
        if metric in res:
            return res[metric]
        return float("nan")

    # If use_training_only is True, skip test_results and only use top-level metrics
    if (
        not use_training_only
        and "test_results" in res
        and metric in res["test_results"]
    ):
        return res["test_results"][metric]
    if metric in res:
        return res[metric]
    return float("nan")


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Compute standard classification metrics from true and predicted labels.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing accuracy, balanced_accuracy, precision, recall, f1, and confusion_matrix
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def compute_roc_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """
    Compute ROC curve metrics from true labels and probability scores.

    Args:
        y_true: True labels
        y_proba: Probability scores for the positive class

    Returns:
        Dictionary containing roc_auc, fpr, and tpr
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc_val = auc(fpr, tpr)
    return {
        "roc_auc": roc_auc_val,
        "fpr": fpr,
        "tpr": tpr,
    }


def truncate_to_match_length(
    longer_array: np.ndarray, target_length: int, from_end: bool = True
) -> np.ndarray:
    """
    Truncate an array to match a target length.

    Args:
        longer_array: Array to truncate
        target_length: Desired length
        from_end: If True, take the last N samples. If False, take the first N samples.

    Returns:
        Truncated array

    Raises:
        ValueError: If target_length is greater than array length
    """
    if len(longer_array) < target_length:
        raise ValueError(
            f"Cannot truncate array of length {len(longer_array)} to length {target_length}"
        )
    if from_end:
        return longer_array[-target_length:]
    else:
        return longer_array[:target_length]


def has_roc_data(results: Dict[str, Any]) -> bool:
    """
    Check if results dictionary contains ROC curve data.

    Args:
        results: Results dictionary

    Returns:
        True if all required ROC keys are present, False otherwise
    """
    return all(key in results for key in ["fpr", "tpr", "roc_auc"])


def reevaluate_against_history(res: Dict[str, Any], pred_res: Dict[str, Any]) -> bool:
    """
    Re-evaluate forecast results against historical predictions.
    Modifies `res` in-place. Returns True on success, False on failure.

    Forecast predictions are limited to the future time window (m seconds),
    while history predictions cover the full trial. We truncate history predictions
    to match the forecast window by taking the last N samples where N is the
    length of forecast predictions.
    """
    if "y_pred" not in pred_res or "y_pred" not in res:
        return False

    history_preds = pred_res["y_pred"]
    y_pred = res["y_pred"]

    if len(history_preds) > len(y_pred):
        history_preds = truncate_to_match_length(
            history_preds, len(y_pred), from_end=True
        )
    elif len(history_preds) < len(y_pred):
        return False

    y_true = history_preds

    # Compute standard classification metrics
    metrics = compute_classification_metrics(y_true, y_pred)
    # Update metrics, overwriting existing ones
    for key, value in metrics.items():
        res[key] = value

    # Extract y_proba from test_results before removing it, or check top level
    # y_proba is needed for ROC metrics computation
    y_proba = None
    if "y_proba" in res:
        y_proba = res["y_proba"]
    elif "test_results" in res and "y_proba" in res["test_results"]:
        y_proba = res["test_results"]["y_proba"]

    # Compute ROC metrics if probability scores are available
    if y_proba is not None:
        # Truncate y_proba to match the length if needed
        if len(y_proba) > len(y_true):
            y_proba = truncate_to_match_length(y_proba, len(y_true), from_end=True)

        if len(y_proba) == len(y_true):
            roc_metrics = compute_roc_metrics(y_true, y_proba)
            # Update ROC metrics, overwriting existing ones
            for key, value in roc_metrics.items():
                res[key] = value

    # Update best_cv_score to use the new balanced_accuracy
    res["best_cv_score"] = res["balanced_accuracy"]

    # Remove test_results since we're evaluating against history, not test set
    res.pop("test_results", None)

    # Also update y_true to reflect that we're using history predictions as ground truth
    res["y_true"] = y_true

    return True


def _apply_line_plot_layout(fig: go.Figure, xaxis_title: str) -> None:
    """Apply shared layout to accuracy-vs-parameter line plots."""
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color=PALETTE.cool_steel,
        annotation_text="Chance",
    )

    # Calculate dynamic y-axis range based on data
    y_min = 0.4
    y_max = 1.0
    if fig.data:
        all_y_values = []
        for trace in fig.data:
            if hasattr(trace, "y") and trace.y is not None:
                y_vals = np.array(trace.y)
                y_vals = y_vals[~np.isnan(y_vals)]
                if len(y_vals) > 0:
                    all_y_values.extend(y_vals.tolist())

        if all_y_values:
            data_min = np.min(all_y_values)
            data_max = np.max(all_y_values)
            # Add padding: 5% below min, 5% above max, but keep within [0, 1] bounds
            padding = (data_max - data_min) * 0.05 if data_max > data_min else 0.05
            y_min = max(0.0, data_min - padding)
            y_max = min(1.0, data_max + padding)
            # Ensure minimum range of 0.2
            if y_max - y_min < 0.2:
                center = (y_max + y_min) / 2
                y_min = max(0.0, center - 0.1)
                y_max = min(1.0, center + 0.1)

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text=xaxis_title,
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            showgrid=True,
            gridcolor="#F0F0F0",
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
        ),
        yaxis=dict(
            title=dict(
                text="Balanced Accuracy",
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            range=[y_min, y_max],
            showgrid=True,
            gridcolor="#F0F0F0",
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
        ),
        template="plotly_white",
        plot_bgcolor="white",
        font=dict(family=PLOT_STYLE.font_family, color=PALETTE.ink_black),
        legend=dict(
            font=dict(size=PLOT_STYLE.tick_label_size, family=PLOT_STYLE.font_family),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E5E5E5",
            borderwidth=1,
            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top",
        ),
        margin=dict(l=60, r=120, t=40, b=60),
        hovermode="closest",
    )


def load_classification_results(
    results_dir: Path, mode: str = "forecast", eval_target: str = "dbs_stim"
) -> Dict[str, Dict[Tuple[float, float], Dict[str, Any]]]:
    """
    Loads all classification results, grouping them by FEATURE SOURCE.
    Returns: { feature_source: { (h, m): result_dict } }
    For predictions: h and m are None (entire trial, no h/m parameters)
    For forecasts: h and m are the actual values from directory names
    """
    all_results = {}
    hm_pattern = re.compile(r"^h([\d.]+)_m([\d.]+)$")

    # For predictions: check directly in results_dir (no h/m subdirectories)
    if mode == "prediction":
        pattern_str = f"*_{mode}.pkl"
        pkl_files = list(results_dir.glob(pattern_str))

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
                # Predictions use entire trial, so h/m are None
                all_results[feature_source][(None, None)] = res
            except Exception as e:
                st.warning(f"Failed to load {pkl_file}: {e}")

    else:
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

                    # Always create a deep copy to avoid modifying the original loaded results
                    # This ensures that switching between eval_targets doesn't affect each other
                    res = copy.deepcopy(res)

                    if eval_target == "history_label" and mode == "forecast":
                        # Predictions are saved in the parent directory (run_ts), not in h/m subdirectories
                        pred_file = pkl_file.parent.parent / pkl_file.name.replace(
                            "_forecast", "_prediction"
                        )
                        if pred_file.exists():
                            try:
                                with open(pred_file, "rb") as fp:
                                    pred_res = pickle.load(fp)
                                if not reevaluate_against_history(res, pred_res):
                                    st.warning(
                                        f"Failed to re-evaluate against history for {pkl_file.name}"
                                    )
                                # Re-evaluation successful - no need to show info message for every file
                            except Exception as e:
                                st.warning(
                                    f"Failed to load history prediction for {pkl_file.name}: {e}"
                                )
                        else:
                            # Prediction file doesn't exist for this feature source
                            # This can happen if predictions were only run for a subset of feature sources
                            # Skip re-evaluation but still include the forecast results
                            # (they just won't have history_label evaluation)
                            pass
                    # For eval_target == "dbs_stim", use the original results as-is (already loaded from disk)

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

    # Extract unique h and m values (filter out None for predictions)
    h_values = sorted(set(hm[0] for hm in results.keys() if hm[0] is not None))
    m_values = sorted(set(hm[1] for hm in results.keys() if hm[1] is not None))

    if not h_values or not m_values:
        return go.Figure()

    # Build the z-matrix
    z_matrix = np.full((len(m_values), len(h_values)), np.nan)

    for (h, m), res in results.items():
        if h is None or m is None:
            continue
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
        margin=dict(l=60, r=100, t=20, b=60),
    )

    return fig


def create_line_plot_by_history(
    all_results: Dict[str, Dict[Tuple[float, float], Dict[str, Any]]],
    metric: str = "balanced_accuracy",
    use_training_only: bool = False,
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
        h_values = sorted(set(hm[0] for hm in results.keys() if hm[0] is not None))
        m_values = sorted(set(hm[1] for hm in results.keys() if hm[1] is not None))

        for m in m_values:
            y_vals = []
            x_vals = []
            for h in h_values:
                if (h, m) in results:
                    y_vals.append(
                        resolve_metric(
                            results[(h, m)], metric, use_training_only=use_training_only
                        )
                    )
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

    _apply_line_plot_layout(fig, "History h (seconds)")
    return fig


def create_line_plot_by_future(
    all_results: Dict[str, Dict[Tuple[float, float], Dict[str, Any]]],
    metric: str = "balanced_accuracy",
    use_training_only: bool = False,
) -> go.Figure:
    if not all_results:
        return go.Figure()

    fig = go.Figure()
    colors = px.colors.qualitative.Safe

    color_idx = 0
    for feature_source, results in all_results.items():
        h_values = sorted(set(hm[0] for hm in results.keys() if hm[0] is not None))
        m_values = sorted(set(hm[1] for hm in results.keys() if hm[1] is not None))

        for h in h_values:
            y_vals = []
            x_vals = []
            for m in m_values:
                if (h, m) in results:
                    y_vals.append(
                        resolve_metric(
                            results[(h, m)], metric, use_training_only=use_training_only
                        )
                    )
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

    _apply_line_plot_layout(fig, "Forecast Horizon m (seconds)")
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
        margin=dict(l=60, r=60, t=20, b=60),
    )

    return fig


def create_summary_table(
    results: Dict[Tuple[float, float], Dict[str, Any]],
    include_hm: bool = True,
) -> List[Dict[str, Any]]:
    """Create a summary table of all h/m results."""
    rows = []
    for (h, m), res in sorted(results.items()):
        row = {}

        if include_hm:
            row["h (s)"] = h if h is not None else "N/A"
            row["m (s)"] = m if m is not None else "N/A"

        row["CV Score"] = res.get("best_cv_score", np.nan)
        row["Balanced Acc"] = res.get("balanced_accuracy", np.nan)

        if "test_results" in res:
            row["Test Acc"] = res["test_results"].get("balanced_accuracy", np.nan)

        if "permutation_test" in res:
            row["p-value"] = res["permutation_test"].get("pvalue", np.nan)

        rows.append(row)

    return rows
