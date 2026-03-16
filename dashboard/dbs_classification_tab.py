import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
import pickle
import pandas as pd
import re
import collections

from utils.classification import (
    prepare_epoched_data,
    run_grid_search_cv,
    evaluate_on_test_set,
)
from dashboard.backbone import (
    PALETTE,
    PLOT_STYLE,
    PLOT_COLOR,
)
from dashboard.subtabs import (
    list_variants,
    list_run_timestamps,
    config_for_variant,
    load_classification_results,
    create_line_plot_by_history,
    create_line_plot_by_future,
    create_heatmap_figure,
    create_summary_table,
    reevaluate_against_history,
    has_roc_data,
)
from sklearn.metrics import confusion_matrix
from utils.classification import (
    load_all_splits,
    load_precomputed_results,
)


def save_classification_results(results: Dict[str, Any], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)


def load_single_result(save_path: Path) -> Optional[Dict[str, Any]]:
    if save_path.exists():
        with open(save_path, "rb") as f:
            try:
                return pickle.load(f)
            except (ImportError, AttributeError, ModuleNotFoundError) as e:
                st.error(f"Failed to load result: {e}")
                return None
    return None


def render_metrics_row(results: Dict[str, Any], prefix: str = "") -> None:
    cols = st.columns(5)
    metrics = [
        ("Accuracy", results.get("accuracy", 0)),
        ("Balanced Acc", results.get("balanced_accuracy", 0)),
        ("Precision", results.get("precision", 0)),
        ("Recall", results.get("recall", 0)),
        ("F1", results.get("f1", 0)),
    ]
    # Add Mean CV Score if available (top-level only)
    if "best_cv_score" in results and "CV " in prefix:
        metrics.insert(0, ("Mean CV Score", results["best_cv_score"]))

    for col, (name, val) in zip(cols, metrics):
        col.metric(f"{prefix}{name}", f"{val:.4f}")


def render_confusion_matrix(cm: np.ndarray, key: str) -> None:

    cm_reordered = np.array([[cm[1, 1], cm[1, 0]], [cm[0, 1], cm[0, 0]]])

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_reordered,
            x=["P", "N"],
            y=["P", "N"],
            colorscale="Blues",
            text=cm_reordered,
            texttemplate="%{text}",
            textfont={"size": 20},
            showscale=True,
        )
    )

    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["TP", "FN"],
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["TP", "FP"],
            autorange="reversed",  # Reverse Y-axis so TP is at top
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"cm_{key}")


def render_roc_curve(results: Dict[str, Any], key: str) -> None:
    # Check if ROC data exists
    if not has_roc_data(results):
        st.info("ROC curve data not available for this evaluation.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results["fpr"],
            y=results["tpr"],
            name=f"ROC (AUC = {results['roc_auc']:.4f})",
            mode="lines",
            line=dict(
                color=PALETTE.twilight_indigo, width=PLOT_STYLE.line_width_normal
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(
                color=PALETTE.strawberry_red,
                dash="dash",
                width=PLOT_STYLE.line_width_normal,
            ),
        )
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="False Positive Rate (FPR)",
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            range=[0, 1],
            showgrid=True,
            gridcolor="#F0F0F0",
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            dtick=0.2,
            constrain="domain",
        ),
        yaxis=dict(
            title=dict(
                text="True Positive Rate (TPR)",
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            range=[0, 1],
            showgrid=True,
            gridcolor="#F0F0F0",
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            dtick=0.2,
            constrain="domain",
        ),
        height=400,
        width=None,
        template="plotly_white",
        plot_bgcolor="white",
        font=dict(
            family=PLOT_STYLE.font_family,
            size=PLOT_STYLE.tick_label_size,
            color="black",
        ),
        legend=dict(
            font=dict(size=10, family=PLOT_STYLE.font_family),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E5E5E5",
            borderwidth=1,
        ),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"roc_{key}")
    st.caption("Receiver Operating Characteristic (ROC)")


def render_results_view(results: Dict[str, Any], key: str) -> None:
    if "best_params" in results:
        st.markdown("#### Best Hyperparameters")
        st.json(results["best_params"])

    st.markdown("#### Cross-Validation Results")
    render_metrics_row(results, "CV ")

    col1, col2 = st.columns(2)
    with col1:
        render_confusion_matrix(results["confusion_matrix"], f"cv_{key}")
    with col2:
        # Only render ROC curve if the data exists
        if has_roc_data(results):
            render_roc_curve(results, f"cv_{key}")
        else:
            st.info("ROC curve not available for this evaluation.")

    if "permutation_test" in results:
        st.markdown("#### Permutation Test")
        p_res = results["permutation_test"]
        p_cols = st.columns(3)
        p_cols[0].metric("Observed Score", f"{p_res['score']:.4f}")
        p_cols[1].metric("p-value", f"{p_res['pvalue']:.4f}")
        p_cols[2].metric("Permutations", f"{p_res.get('n_permutations', 'N/A')}")

    if "test_results" in results:
        st.markdown("---")
        st.markdown("#### Test Set Results (Held Out)")
        test_res = results["test_results"]
        render_metrics_row(test_res, "Test ")

        col1, col2 = st.columns(2)
        with col1:
            render_confusion_matrix(test_res["confusion_matrix"], f"test_{key}")
        with col2:
            # Only render ROC curve if the data exists
            if has_roc_data(test_res):
                render_roc_curve(test_res, f"test_{key}")
            else:
                st.info("ROC curve not available for this evaluation.")


def render_fold_results(fold_results: list, key: str) -> None:
    st.markdown("#### Per-Fold Results (Chronological)")

    df = pd.DataFrame(fold_results)
    df["Train Range"] = df.apply(
        lambda r: f"[{r['train_indices'][0]}:{r['train_indices'][1]}]", axis=1
    )
    df["Val Range"] = df.apply(
        lambda r: f"[{r['val_indices'][0]}:{r['val_indices'][1]}]", axis=1
    )
    df["Train ON/OFF"] = df.apply(
        lambda r: f"{r['n_on_train']}/{r['n_off_train']}", axis=1
    )
    df["Val ON/OFF"] = df.apply(lambda r: f"{r['n_on_val']}/{r['n_off_val']}", axis=1)

    display_df = df[
        [
            "fold",
            "Train Range",
            "Val Range",
            "Train ON/OFF",
            "Val ON/OFF",
            "accuracy",
            "balanced_accuracy",
        ]
    ].copy()
    display_df.columns = [
        "Fold",
        "Train Range",
        "Val Range",
        "Train ON/OFF",
        "Val ON/OFF",
        "Acc",
        "Bal Acc",
    ]

    st.dataframe(display_df, use_container_width=True, key=f"tbl_fold_perf_{key}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["fold"],
            y=df["accuracy"],
            mode="lines+markers",
            name="Accuracy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["fold"],
            y=df["balanced_accuracy"],
            mode="lines+markers",
            name="Balanced Accuracy",
        )
    )
    fig.update_layout(
        xaxis_title="Fold",
        yaxis_title="Score",
        height=350,
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=50, r=20, t=20, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"fold_perf_{key}")
    st.caption("Cross-Validation Performance Across Folds")


def render_classifier_section(
    clf_name: str,
    feature_source: str,
    epoch_params: str,
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    groups_trainval: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    results_dir: Path,
    n_splits: int,
    mode: str = "prediction",
    forecast_horizon_sec: Optional[float] = None,
) -> None:
    key_base = f"{clf_name}_{feature_source}_{mode}_{epoch_params}".replace(
        " ", "_"
    ).replace(".", "p")

    cache_path = results_dir / f"{key_base}.pkl"

    if not cache_path.exists() and forecast_horizon_sec:
        key_base_with_fh = key_base + f"_fh{forecast_horizon_sec}s".replace(".", "p")
        cache_path_with_fh = results_dir / f"{key_base_with_fh}.pkl"
        if cache_path_with_fh.exists():
            key_base = key_base_with_fh
            cache_path = cache_path_with_fh

    st.markdown(f"### {clf_name}")

    cached = load_single_result(cache_path)
    results = st.session_state.get(f"results_{key_base}", cached)

    if results:
        st.success("Precomputed results loaded")
        render_results_view(results, key_base)
    else:
        st.warning("No precomputed results found. Run the classification script first.")


def render_classification_results(
    variant_dir: Path,
    run_ts: str,
    mode: str,
    eval_target: str = "dbs_stim",
) -> None:

    results_dir = variant_dir / run_ts

    if not results_dir.exists():
        st.warning(
            f"No classification directory found at {results_dir.relative_to(variant_dir.parent.parent)}. Run the classification script first."
        )
        return

    # Always search recursively to find all possible results
    pattern = f"*_{mode}.pkl"
    result_files = list(results_dir.rglob(pattern))

    if not result_files:
        st.info(
            f"No {mode} classification results found. Run the classification script to generate results."
        )
        return

    # Parse all files into a structured list
    hm_pattern = re.compile(r"h([\d.]+)_m([\d.]+)")
    file_records = []

    for f in result_files:
        filename = f.stem
        parts = filename.split("_")

        # Extract feature source
        suffixes = {"prediction", "forecast", "flipped"}
        stop_idx = len(parts)
        for i, p in enumerate(parts):
            if p in suffixes:
                stop_idx = i
                break
        feature_source = "_".join(parts[1:stop_idx])

        # Extract h and m
        match = hm_pattern.search(f.parent.name)
        if match:
            h, m = float(match.group(1)), float(match.group(2))
        else:
            h, m = None, None  # Standard/Top-level

        file_records.append(
            {"file": f, "feature": feature_source, "h": h, "m": m, "clf": parts[0]}
        )

    # Selectors
    st.markdown("#### Filter Detailed Results")

    # Get unique values for selectors
    all_feats = sorted(set(r["feature"] for r in file_records))
    all_h = sorted(set(r["h"] for r in file_records if r["h"] is not None))
    all_m = sorted(set(r["m"] for r in file_records if r["m"] is not None))

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_feat = st.selectbox(
            "Feature Source", options=all_feats, key=f"det_feat_{mode}_{run_ts}"
        )

    if mode == "prediction":
        sel_h, sel_m = None, None
    elif all_h or all_m:
        with col2:
            h_options = all_h + (
                ["Standard"] if any(r["h"] is None for r in file_records) else []
            )
            sel_h = st.selectbox(
                "History h (s)", options=h_options, key=f"det_h_{mode}_{run_ts}"
            )
        with col3:
            m_options = all_m + (
                ["Standard"] if any(r["m"] is None for r in file_records) else []
            )
            sel_m = st.selectbox(
                "Horizon m (s)", options=m_options, key=f"det_m_{mode}_{run_ts}"
            )
    else:
        sel_h, sel_m = "Standard", "Standard"

    def matches(r):
        feat_match = r["feature"] == sel_feat
        if mode == "prediction":
            return feat_match
        h_val = r["h"] if r["h"] is not None else "Standard"
        m_val = r["m"] if r["m"] is not None else "Standard"
        return feat_match and h_val == sel_h and m_val == sel_m

    matched_records = [r for r in file_records if matches(r)]

    if not matched_records:
        st.warning(f"No results match the selected filters.")
        return

    for r in matched_records:
        result_file = r["file"]
        display_name = f"{r['clf']} - {r['feature']} features"

        st.markdown(f"### {display_name}")
        if mode != "prediction" and r["h"] is not None:
            st.caption(f"Window: h={r['h']}s, m={r['m']}s")

        results = load_single_result(result_file)
        if results:
            if eval_target == "history_label":
                # Predictions are saved in run_ts directory, not in h/m subdirectories
                if mode == "forecast":
                    pred_file = result_file.parent.parent / result_file.name.replace(
                        "_forecast", "_prediction"
                    )
                else:
                    pred_file = result_file.parent / result_file.name.replace(
                        "_forecast", "_prediction"
                    )
                if pred_file.exists():
                    pred_res = load_single_result(pred_file)
                    if pred_res:
                        if reevaluate_against_history(results, pred_res):
                            n = len(results["y_pred"])
                            st.info(
                                f"Evaluating {n} forecast samples against historical predictions."
                            )
                        else:
                            st.error(
                                "Sample size mismatch between history and forecast predictions."
                            )
                    else:
                        st.warning(
                            "Could not load history predictions for this configuration."
                        )
                else:
                    st.warning(f"Matching prediction file not found: {pred_file.name}")

            render_results_view(results, result_file.stem)

        st.markdown("---")


def render_classification_from_predictions(variant_dir: Path, run_ts: str) -> None:
    results_dir = variant_dir / run_ts
    if results_dir.exists():
        all_results = load_classification_results(results_dir, mode="prediction")
        n_configs = sum(len(res) for res in all_results.values())
        if n_configs > 1 or len(all_results) > 1:
            st.markdown("## Prediction Performance Summary")
            render_prediction_summary(all_results, f"pred_{run_ts}")
            st.markdown("---")

    render_classification_results(variant_dir, run_ts, mode="prediction")


def _render_prediction_feature_summary(
    feature_source: str,
    feat_results: Dict[Tuple[float, float], Dict[str, Any]],
    key_prefix: str,
) -> None:
    """Render summary table for a single feature source in predictions."""
    st.markdown(f"### Feature Source: {feature_source}")

    # Show summary table
    summary_rows = create_summary_table(feat_results, include_hm=False)
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        render_styled_table(df, key=f"pred_summary_{feature_source}_{key_prefix}")

        # Create caption with metrics from table
        row = summary_rows[0]
        metrics_parts = []
        if "CV Score" in row and not pd.isna(row["CV Score"]):
            metrics_parts.append(f"CV Score: {row['CV Score']:.4f}")
        if "Balanced Acc" in row and not pd.isna(row["Balanced Acc"]):
            metrics_parts.append(f"Balanced Accuracy: {row['Balanced Acc']:.4f}")
        if "Test Acc" in row and not pd.isna(row["Test Acc"]):
            metrics_parts.append(f"Test Accuracy: {row['Test Acc']:.4f}")
        if "p-value" in row and not pd.isna(row["p-value"]):
            metrics_parts.append(f"p-value: {row['p-value']:.4f}")

        if metrics_parts:
            st.caption(" | ".join(metrics_parts))

    st.markdown("---")


def _create_roc_heatmap(
    results: Dict[Tuple[float, float], Dict[str, Any]],
    feature_source: str,
) -> go.Figure:
    """Create a heatmap showing ROC AUC across h and m values."""
    if not results:
        return go.Figure()

    # Extract unique h and m values (filter out None for predictions)
    h_values = sorted(set(hm[0] for hm in results.keys() if hm[0] is not None))
    m_values = sorted(set(hm[1] for hm in results.keys() if hm[1] is not None))

    if not h_values or not m_values:
        return go.Figure()

    # Build the z-matrix for ROC AUC
    z_matrix = np.full((len(m_values), len(h_values)), np.nan)

    for (h, m), res in results.items():
        if h is None or m is None:
            continue
        h_idx = h_values.index(h)
        m_idx = m_values.index(m)

        # Get ROC AUC from test_results if available, otherwise from top level
        roc_auc = None
        if "test_results" in res and "roc_auc" in res["test_results"]:
            roc_auc = res["test_results"]["roc_auc"]
        elif "roc_auc" in res:
            roc_auc = res["roc_auc"]

        if roc_auc is not None:
            z_matrix[m_idx, h_idx] = roc_auc

    # Find best cell
    if not np.isnan(z_matrix).all():
        best_idx = np.nanargmax(z_matrix)
        best_m_idx, best_h_idx = np.unravel_index(best_idx, z_matrix.shape)
        best_h = h_values[best_h_idx]
        best_m = m_values[best_m_idx]
    else:
        best_m_idx = best_h_idx = None

    # Create heatmap
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z_matrix,
            x=[f"{h:.1f}" for h in h_values],
            y=[f"{m:.1f}" for m in m_values],
            colorscale=[
                [0.0, "white"],
                [1.0, PALETTE.strawberry_red],
            ],
            colorbar=dict(
                title=dict(
                    text="ROC<br>AUC",
                    font=dict(
                        size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                    ),
                ),
                tickfont=dict(size=PLOT_STYLE.tick_label_size),
            ),
            hovertemplate="h=%{x}s, m=%{y}s<br>ROC AUC=%{z:.3f}<extra></extra>",
            zmin=0.5,  # Chance level
            zmax=1.0,
        )
    )

    # Add annotation for best cell
    if best_m_idx is not None and best_h_idx is not None:
        fig.add_annotation(
            x=f"{h_values[best_h_idx]:.1f}",
            y=f"{m_values[best_m_idx]:.1f}",
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


def _render_balanced_acc_heatmap_subplot(
    all_results: Dict[str, Dict[Tuple[float, float], Dict[str, Any]]],
    key_prefix: str,
) -> None:
    """Create a subplot with Balanced Accuracy heatmaps for all feature sources."""
    st.markdown("### Balanced Accuracy Heatmaps")

    n_features = len(all_results)
    if n_features == 0:
        return

    # Determine subplot layout: try to make it roughly square
    n_cols = int(np.ceil(np.sqrt(n_features)))
    n_rows = int(np.ceil(n_features / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[fs for fs in all_results.keys()],
        horizontal_spacing=0.15,
        vertical_spacing=0.12,
    )

    for idx, (feature_source, feat_results) in enumerate(all_results.items()):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        # Extract unique h and m values
        h_values = sorted(set(hm[0] for hm in feat_results.keys() if hm[0] is not None))
        m_values = sorted(set(hm[1] for hm in feat_results.keys() if hm[1] is not None))

        if not h_values or not m_values:
            continue

        # Build the z-matrix for Balanced Accuracy
        z_matrix = np.full((len(m_values), len(h_values)), np.nan)

        for (h, m), res in feat_results.items():
            if h is None or m is None:
                continue
            h_idx = h_values.index(h)
            m_idx = m_values.index(m)

            # Get Balanced Accuracy from test_results if available, otherwise from top level
            balanced_acc = None
            if "test_results" in res and "balanced_accuracy" in res["test_results"]:
                balanced_acc = res["test_results"]["balanced_accuracy"]
            elif "balanced_accuracy" in res:
                balanced_acc = res["balanced_accuracy"]

            if balanced_acc is not None:
                z_matrix[m_idx, h_idx] = balanced_acc

        # Add heatmap to subplot
        fig.add_trace(
            go.Heatmap(
                z=z_matrix,
                x=[f"{h:.1f}" for h in h_values],
                y=[f"{m:.1f}" for m in m_values],
                colorscale=[
                    [0.0, "white"],
                    [1.0, PALETTE.strawberry_red],
                ],
                hovertemplate="h=%{x}s, m=%{y}s<br>Balanced Acc=%{z:.3f}<extra></extra>",
                zmin=0.5,
                zmax=1.0,
                showscale=(idx == 0),  # Only show colorbar for first subplot
                colorbar=(
                    dict(
                        title=dict(
                            text="Balanced<br>Accuracy",
                            font=dict(
                                size=PLOT_STYLE.axis_label_size,
                                family=PLOT_STYLE.font_family,
                            ),
                        ),
                        tickfont=dict(size=PLOT_STYLE.tick_label_size),
                        len=0.6,
                    )
                    if idx == 0
                    else None
                ),
            ),
            row=row,
            col=col,
        )

        # Update axes for this subplot
        fig.update_xaxes(
            title_text="h (s)" if row == n_rows else "",
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="m (s)" if col == 1 else "",
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            row=row,
            col=col,
        )

    # Update overall layout
    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family, color=PALETTE.ink_black),
        height=300 * n_rows,
        margin=dict(l=60, r=60, t=60, b=60),
        showlegend=False,
    )

    st.plotly_chart(
        fig, use_container_width=True, key=f"balanced_acc_heatmap_subplot_{key_prefix}"
    )
    st.caption(
        "Balanced Accuracy across History (h) and Forecast Horizon (m) for all feature sources"
    )
    st.markdown("---")


def _render_roc_heatmap_subplot(
    all_results: Dict[str, Dict[Tuple[float, float], Dict[str, Any]]],
    key_prefix: str,
) -> None:
    """Create a subplot with ROC AUC heatmaps for all feature sources."""
    # Check if any feature has ROC data
    has_any_roc = any(
        any(has_roc_data(res) for res in feat_results.values())
        for feat_results in all_results.values()
    )

    if not has_any_roc:
        return

    st.markdown("### ROC AUC Heatmaps")

    n_features = len(all_results)
    if n_features == 0:
        return

    # Determine subplot layout: try to make it roughly square
    n_cols = int(np.ceil(np.sqrt(n_features)))
    n_rows = int(np.ceil(n_features / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[fs for fs in all_results.keys()],
        horizontal_spacing=0.15,
        vertical_spacing=0.12,
    )

    for idx, (feature_source, feat_results) in enumerate(all_results.items()):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        # Extract unique h and m values
        h_values = sorted(set(hm[0] for hm in feat_results.keys() if hm[0] is not None))
        m_values = sorted(set(hm[1] for hm in feat_results.keys() if hm[1] is not None))

        if not h_values or not m_values:
            continue

        # Build the z-matrix for ROC AUC
        z_matrix = np.full((len(m_values), len(h_values)), np.nan)

        for (h, m), res in feat_results.items():
            if h is None or m is None:
                continue
            h_idx = h_values.index(h)
            m_idx = m_values.index(m)

            # Get ROC AUC from test_results if available, otherwise from top level
            roc_auc = None
            if "test_results" in res and "roc_auc" in res["test_results"]:
                roc_auc = res["test_results"]["roc_auc"]
            elif "roc_auc" in res:
                roc_auc = res["roc_auc"]

            if roc_auc is not None:
                z_matrix[m_idx, h_idx] = roc_auc

        # Add heatmap to subplot
        fig.add_trace(
            go.Heatmap(
                z=z_matrix,
                x=[f"{h:.1f}" for h in h_values],
                y=[f"{m:.1f}" for m in m_values],
                colorscale=[
                    [0.0, "white"],
                    [1.0, PALETTE.strawberry_red],
                ],
                hovertemplate="h=%{x}s, m=%{y}s<br>ROC AUC=%{z:.3f}<extra></extra>",
                zmin=0.5,
                zmax=1.0,
                showscale=(idx == 0),  # Only show colorbar for first subplot
                colorbar=(
                    dict(
                        title=dict(
                            text="ROC<br>AUC",
                            font=dict(
                                size=PLOT_STYLE.axis_label_size,
                                family=PLOT_STYLE.font_family,
                            ),
                        ),
                        tickfont=dict(size=PLOT_STYLE.tick_label_size),
                        len=0.6,
                    )
                    if idx == 0
                    else None
                ),
            ),
            row=row,
            col=col,
        )

        # Update axes for this subplot
        fig.update_xaxes(
            title_text="h (s)" if row == n_rows else "",
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="m (s)" if col == 1 else "",
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            row=row,
            col=col,
        )

    # Update overall layout
    fig.update_layout(
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family, color=PALETTE.ink_black),
        height=300 * n_rows,
        margin=dict(l=60, r=60, t=60, b=60),
        showlegend=False,
    )

    st.plotly_chart(
        fig, use_container_width=True, key=f"roc_heatmap_subplot_{key_prefix}"
    )
    st.caption(
        "ROC AUC across History (h) and Forecast Horizon (m) for all feature sources"
    )
    st.markdown("---")


def _create_roc_heatmap(
    results: Dict[Tuple[float, float], Dict[str, Any]],
    feature_source: str,
) -> go.Figure:
    """Create a heatmap showing ROC AUC across h and m values."""
    if not results:
        return go.Figure()

    # Extract unique h and m values (filter out None for predictions)
    h_values = sorted(set(hm[0] for hm in results.keys() if hm[0] is not None))
    m_values = sorted(set(hm[1] for hm in results.keys() if hm[1] is not None))

    if not h_values or not m_values:
        return go.Figure()

    # Build the z-matrix for ROC AUC
    z_matrix = np.full((len(m_values), len(h_values)), np.nan)

    for (h, m), res in results.items():
        if h is None or m is None:
            continue
        h_idx = h_values.index(h)
        m_idx = m_values.index(m)

        # Get ROC AUC from test_results if available, otherwise from top level
        roc_auc = None
        if "test_results" in res and "roc_auc" in res["test_results"]:
            roc_auc = res["test_results"]["roc_auc"]
        elif "roc_auc" in res:
            roc_auc = res["roc_auc"]

        if roc_auc is not None:
            z_matrix[m_idx, h_idx] = roc_auc

    # Find best cell
    if not np.isnan(z_matrix).all():
        best_idx = np.nanargmax(z_matrix)
        best_m_idx, best_h_idx = np.unravel_index(best_idx, z_matrix.shape)
        best_h = h_values[best_h_idx]
        best_m = m_values[best_m_idx]
    else:
        best_m_idx = best_h_idx = None

    # Create heatmap
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z_matrix,
            x=[f"{h:.1f}" for h in h_values],
            y=[f"{m:.1f}" for m in m_values],
            colorscale=[
                [0.0, "white"],
                [1.0, PALETTE.strawberry_red],
            ],
            colorbar=dict(
                title=dict(
                    text="ROC<br>AUC",
                    font=dict(
                        size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                    ),
                ),
                tickfont=dict(size=PLOT_STYLE.tick_label_size),
            ),
            hovertemplate="h=%{x}s, m=%{y}s<br>ROC AUC=%{z:.3f}<extra></extra>",
            zmin=0.5,  # Chance level
            zmax=1.0,
        )
    )

    # Add annotation for best cell
    if best_m_idx is not None and best_h_idx is not None:
        fig.add_annotation(
            x=f"{h_values[best_h_idx]:.1f}",
            y=f"{m_values[best_m_idx]:.1f}",
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


def _render_forecast_feature_summary(
    feature_source: str,
    feat_results: Dict[Tuple[float, float], Dict[str, Any]],
    key_prefix: str,
) -> None:
    """Render h/m plots for a single feature source in forecasts."""
    st.markdown(f"### Feature Source: {feature_source}")
    single_feat_results = {feature_source: feat_results}
    metric = "balanced_accuracy"

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            create_line_plot_by_history(
                single_feat_results, metric=metric, use_training_only=False
            ),
            use_container_width=True,
            key=f"h_plot_{feature_source}_{key_prefix}",
        )
        st.caption("Accuracy vs History Length")
    with col2:
        st.plotly_chart(
            create_line_plot_by_future(
                single_feat_results, metric=metric, use_training_only=False
            ),
            use_container_width=True,
            key=f"m_plot_{feature_source}_{key_prefix}",
        )
        st.caption("Accuracy vs Forecast Horizon")
    st.markdown("---")


def render_prediction_summary(
    all_results: Dict[str, Dict[Tuple[float, float], Dict[str, Any]]], key_prefix: str
) -> None:
    """Render summary for predictions."""
    if not all_results:
        return

    # Test results are computed on-the-go when test data exists
    # If test results don't exist, we'll just skip showing them in the summary

    # Create combined ROC plot with all feature sources
    has_any_roc = False
    fig = go.Figure()

    # Add ROC curves for each feature source
    colors = px.colors.qualitative.Plotly

    for idx, (feature_source, feat_results) in enumerate(all_results.items()):
        # Get the result (predictions have only one result with key (None, None))
        result = None
        for key, res in feat_results.items():
            result = res
            break

        if result and has_roc_data(result):
            has_any_roc = True
            fig.add_trace(
                go.Scatter(
                    x=result["fpr"],
                    y=result["tpr"],
                    name=f"{feature_source} (AUC = {result['roc_auc']:.4f})",
                    mode="lines",
                    line=dict(
                        color=colors[idx % len(colors)],
                        width=PLOT_STYLE.line_width_normal,
                    ),
                )
            )

    # Add random baseline line
    if has_any_roc:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(
                    color=PALETTE.strawberry_red,
                    dash="dash",
                    width=PLOT_STYLE.line_width_normal,
                ),
            )
        )

        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text="False Positive Rate (FPR)",
                    font=dict(
                        size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                    ),
                ),
                tickfont=dict(size=PLOT_STYLE.tick_label_size),
                range=[0, 1],
                showgrid=True,
                gridcolor="#F0F0F0",
                showline=True,
                linecolor="black",
                linewidth=1,
                mirror=True,
                dtick=0.2,
                constrain="domain",
            ),
            yaxis=dict(
                title=dict(
                    text="True Positive Rate (TPR)",
                    font=dict(
                        size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                    ),
                ),
                tickfont=dict(size=PLOT_STYLE.tick_label_size),
                range=[0, 1],
                showgrid=True,
                gridcolor="#F0F0F0",
                showline=True,
                linecolor="black",
                linewidth=1,
                mirror=True,
                dtick=0.2,
                constrain="domain",
            ),
            height=400,
            width=None,
            template="plotly_white",
            plot_bgcolor="white",
            font=dict(
                family=PLOT_STYLE.font_family,
                size=PLOT_STYLE.tick_label_size,
                color="black",
            ),
            legend=dict(
                font=dict(size=10, family=PLOT_STYLE.font_family),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#E5E5E5",
                borderwidth=1,
            ),
            margin=dict(l=60, r=40, t=60, b=60),
        )
        st.plotly_chart(
            fig, use_container_width=True, key=f"pred_roc_combined_{key_prefix}"
        )

        # Create caption with aggregated metrics from all feature sources
        all_metrics = []
        for feature_source, feat_results in all_results.items():
            summary_rows = create_summary_table(feat_results, include_hm=False)
            if summary_rows:
                row = summary_rows[0]
                feat_metrics = [f"{feature_source}:"]
                if "CV Score" in row and not pd.isna(row["CV Score"]):
                    feat_metrics.append(f"CV={row['CV Score']:.4f}")
                if "Balanced Acc" in row and not pd.isna(row["Balanced Acc"]):
                    feat_metrics.append(f"BalAcc={row['Balanced Acc']:.4f}")
                if "Test Acc" in row and not pd.isna(row["Test Acc"]):
                    feat_metrics.append(f"Test={row['Test Acc']:.4f}")
                if "p-value" in row and not pd.isna(row["p-value"]):
                    feat_metrics.append(f"p={row['p-value']:.4f}")
                all_metrics.append(" ".join(feat_metrics))

        if all_metrics:
            st.caption(" | ".join(all_metrics))
        else:
            st.caption("ROC curves for all feature sources")


def render_forecast_summary(
    all_results: Dict[str, Dict[Tuple[float, float], Dict[str, Any]]], key_prefix: str
) -> None:
    """Render summary for forecasts with h/m plots."""
    if not all_results:
        return

    # Test results are computed on-the-go when test data exists
    # If test results don't exist, we'll just skip showing them in the summary

    # Create Balanced Accuracy heatmap subplot for all feature sources
    _render_balanced_acc_heatmap_subplot(all_results, key_prefix)

    # Create ROC AUC heatmap subplot for all feature sources
    _render_roc_heatmap_subplot(all_results, key_prefix)

    # Render individual feature summaries
    for feature_source, feat_results in all_results.items():
        _render_forecast_feature_summary(feature_source, feat_results, key_prefix)


def render_classification_from_forecasts(variant_dir: Path, run_ts: str) -> None:
    st.markdown("### Forecast Evaluation Settings")
    eval_target = st.radio(
        "Evaluate Forecast Against:",
        options=["dbs_stim", "history_label"],
        format_func=lambda x: (
            "True Label (dbs_stim)"
            if x == "dbs_stim"
            else "Historical Prediction (history_label)"
        ),
        horizontal=True,
        key=f"eval_target_forecast_{run_ts}",
    )

    results_dir = variant_dir / run_ts
    if results_dir.exists():
        all_results = load_classification_results(
            results_dir, mode="forecast", eval_target=eval_target
        )
        # Show summary if we have multiple windows OR multiple features
        n_configs = sum(len(res) for res in all_results.values())
        if n_configs > 1 or len(all_results) > 1:
            st.markdown("## Forecast Performance Summary")
            render_forecast_summary(all_results, f"forecast_{run_ts}")
            st.markdown("---")

    render_classification_results(
        variant_dir, run_ts, mode="forecast", eval_target=eval_target
    )


def dbs_classification_tab(
    project_root: Path, results_root: Optional[Path] = None
) -> None:
    st.header("DBS ON/OFF Classification")
    RESULTS_ROOT = (
        (results_root / "classification")
        if results_root
        else (project_root / "results" / "classification")
    )

    variants = list_variants(RESULTS_ROOT)
    if len(variants) == 0:
        st.info("No result variants found under results/classification/.")
        return

    variant = st.selectbox("Model variant", options=variants, key="class_variant")
    variant_dir = RESULTS_ROOT / variant
    runs = list_run_timestamps(variant_dir)

    if len(runs) == 0:
        st.info("No runs found for this variant yet. Train a model first.")
        return

    # Default to the latest (most recent) timestamp
    run_ts = st.selectbox(
        "Run timestamp", options=runs, index=len(runs) - 1, key="class_run"
    )
    cfg_path = config_for_variant(project_root, variant)

    if cfg_path is None:
        st.error(f"Config not found for variant '{variant}'.")
        return

    classification_dir = variant_dir / run_ts

    has_flipped_results = False
    hm_dirs = []
    if classification_dir.exists():
        hm_pattern = re.compile(r"^h[\d.]+_m[\d.]+$")
        hm_dirs = [
            d
            for d in classification_dir.iterdir()
            if d.is_dir() and hm_pattern.match(d.name)
        ]
        has_flipped_results = len(hm_dirs) > 0 and "flipped" in variant.lower()

    if classification_dir.exists():
        num_pkl = len(list(classification_dir.glob("*.pkl")))
        num_hm = len(hm_dirs)
        if num_pkl > 0 or num_hm > 0:
            msg_parts = []
            if num_pkl > 0:
                msg_parts.append(f"{num_pkl} standard classification result(s)")
            if num_hm > 0:
                msg_parts.append(f"{num_hm} (h, m) combinations")
            st.success(f"Found {', '.join(msg_parts)}")
        else:
            st.warning(
                "No precomputed results found. Run the classification script first."
            )
    else:
        st.warning(
            "No classification directory found. Run the classification script first."
        )

    st.markdown("---")

    if has_flipped_results:
        st.markdown(
            "## Classification Results on Different History and Forecast Windows"
        )

        current_selection = (variant, run_ts)
        if st.session_state.get("flipped_selection") != current_selection:
            st.session_state["flipped_results"] = None
            st.session_state["flipped_selection"] = current_selection

        if st.button("Load/Refresh Flipped Results"):
            with st.spinner("Loading results..."):
                st.session_state["flipped_results"] = load_classification_results(
                    classification_dir, mode="flipped"
                )

        flipped_results = st.session_state.get("flipped_results")

        if not flipped_results:
            st.info(
                "Click 'Load/Refresh Flipped Results' to visualize the (h, m) history/forecast."
            )
        else:
            st.markdown("### Flipped Performance Summary")
            render_forecast_summary(flipped_results, f"flipped_{run_ts}")

            st.markdown("---")
            st.markdown("### Detailed Flipped Analysis")

            all_feats = sorted(flipped_results.keys())
            h_values = sorted(
                set(k[0] for res in flipped_results.values() for k in res.keys())
            )
            m_values = sorted(
                set(k[1] for res in flipped_results.values() for k in res.keys())
            )

            col_sel1, col_sel2, col_sel3 = st.columns(3)
            with col_sel1:
                sel_feat = st.selectbox(
                    "Select Feature Source", options=all_feats, key="flipped_sel_feat"
                )
            with col_sel2:
                sel_h = st.selectbox(
                    "Select History h (s)", options=h_values, key="flipped_sel_h"
                )
            with col_sel3:
                sel_m = st.selectbox(
                    "Select Forecast Horizon m (s)",
                    options=m_values,
                    key="flipped_sel_m",
                )

            selected_res = flipped_results[sel_feat].get((sel_h, sel_m))
            if selected_res:
                st.markdown(f"#### Results for {sel_feat} - h={sel_h}s, m={sel_m}s")
                render_results_view(selected_res, f"flipped_{sel_feat}_{sel_h}_{sel_m}")
            else:
                st.warning(
                    f"No results found for feature '{sel_feat}' with h={sel_h}s, m={sel_m}s"
                )
    else:
        mode_tabs = st.tabs(["From Predictions", "From Forecasts"])

        with mode_tabs[0]:
            render_classification_from_predictions(variant_dir, run_ts)

        with mode_tabs[1]:
            render_classification_from_forecasts(variant_dir, run_ts)
