import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
import pickle
import pandas as pd
import re
import collections

from utils.classification import (
    CLASSIFIERS,
    prepare_epoched_data,
    run_grid_search_cv,
    evaluate_on_test_set,
)
from dashboard.backbone import (
    PALETTE,
    PLOT_STYLE,
    PLOT_COLOR,
)
from dashboard.subtabs.classification import (
    load_classification_results,
    create_line_plot_by_history,
    create_line_plot_by_future,
    create_heatmap_figure,
    create_summary_table,
)
from dashboard.subtabs import (
    load_precomputed_results,
    list_variants,
    list_run_timestamps,
    config_for_variant,
)


def load_all_splits(
    variant_dir: Path, run_ts: str
) -> Dict[str, Optional[Dict[str, Any]]]:

    splits = {}
    for split_name in ["train", "val", "test"]:
        splits[split_name] = load_precomputed_results(variant_dir, run_ts, split_name)
    return splits


def save_classification_results(results: Dict[str, Any], save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)


def load_single_result(save_path: Path) -> Optional[Dict[str, Any]]:
    # Simple load after MNE upgrade
    if save_path.exists():
        with open(save_path, "rb") as f:
            try:
                return pickle.load(f)
            except (ImportError, AttributeError, ModuleNotFoundError) as e:
                st.error(f"Failed to load result: {e}")
                return None
    return None


def render_metrics_row(results: Dict[str, Any], prefix: str = ""):
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


def render_confusion_matrix(cm: np.ndarray, key: str):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm[::-1, ::-1],
            x=["ON", "OFF"],
            y=["ON", "OFF"],
            colorscale="Burg",
            text=cm[::-1, ::-1],
            texttemplate="%{text}",
            textfont={"size": 20},
            showscale=True,
        )
    )
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis=dict(autorange="reversed"),
        height=350,
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        margin=dict(l=40, r=40, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"cm_{key}")


def render_roc_curve(results: Dict[str, Any], key: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results["fpr"],
            y=results["tpr"],
            name=f"ROC (AUC = {results['roc_auc']:.4f})",
            line=dict(color=PALETTE.twilight_indigo, width=PLOT_STYLE.line_width_normal),
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
        xaxis_title="FP Rate",
        yaxis_title="TP Rate",
        height=350,
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        margin=dict(l=50, r=20, t=10, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"roc_{key}")


def render_results_view(results: Dict[str, Any], key: str):
    if "best_params" in results:
        st.markdown("#### Best Hyperparameters")
        st.json(results["best_params"])

    st.markdown("#### Cross-Validation Results")
    render_metrics_row(results, "CV ")

    col1, col2 = st.columns(2)
    with col1:
        render_confusion_matrix(results["confusion_matrix"], f"cv_{key}")
    with col2:
        render_roc_curve(results, f"cv_{key}")

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
            render_roc_curve(test_res, f"test_{key}")


def render_fold_results(fold_results: list, key: str):
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

    st.dataframe(display_df, use_container_width=True, hide_index=True)

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
        title="Per-Fold Performance",
        xaxis_title="Fold",
        yaxis_title="Score",
        height=300,
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        margin=dict(l=50, r=20, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"fold_perf_{key}")


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
):
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
        st.warning(
            "No precomputed results found. Run the classification script first or click 'Compute Now' below."
        )

    with st.expander("Recompute Classification", expanded=False):
        st.markdown(
            "**Note:** This will recompute the classification. Use the standalone script for batch processing."
        )
        if st.button(f"Compute Now", key=f"gs_{key_base}"):
            with st.spinner("Running grid search with ChronoGroupsSplit CV..."):
                best_params, best_score, cv_results = run_grid_search_cv(
                    clf_name, X_trainval, y_trainval, groups_trainval, n_splits
                )

                if X_test is not None and y_test is not None and len(y_test) > 0:
                    best_pipeline = cv_results.get("best_pipeline")
                    if best_pipeline is not None:
                        test_results = evaluate_on_test_set(
                            best_pipeline, X_test, y_test
                        )
                        cv_results["test_results"] = test_results

                save_classification_results(cv_results, cache_path)
                st.session_state[f"results_{key_base}"] = cv_results
                results = cv_results
                st.rerun()


def render_classification_mode(
    variant_dir: Path,
    run_ts: str,
    mode: str,
):

    results_dir = variant_dir / run_ts / "classification"

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
            
        file_records.append({
            "file": f,
            "feature": feature_source,
            "h": h,
            "m": m,
            "clf": parts[0]
        })

    # Selectors
    st.markdown("#### Filter Detailed Results")
    
    # Get unique values for selectors
    all_feats = sorted(set(r["feature"] for r in file_records))
    all_h = sorted(set(r["h"] for r in file_records if r["h"] is not None))
    all_m = sorted(set(r["m"] for r in file_records if r["m"] is not None))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sel_feat = st.selectbox(
            "Feature Source", 
            options=all_feats, 
            key=f"det_feat_{mode}_{run_ts}"
        )
    
    # Only show h/m selectors if we have labeled windows
    if all_h or all_m:
        with col2:
            # Add "Standard" or "N/A" if there are files without h/m
            h_options = all_h + (["Standard"] if any(r["h"] is None for r in file_records) else [])
            sel_h = st.selectbox(
                "History h (s)", 
                options=h_options, 
                key=f"det_h_{mode}_{run_ts}"
            )
        with col3:
            m_options = all_m + (["Standard"] if any(r["m"] is None for r in file_records) else [])
            sel_m = st.selectbox(
                "Horizon m (s)", 
                options=m_options, 
                key=f"det_m_{mode}_{run_ts}"
            )
    else:
        sel_h, sel_m = "Standard", "Standard"

    # Filter records based on selection
    def matches(r):
        feat_match = r["feature"] == sel_feat
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
        if r["h"] is not None:
            st.caption(f"Window: h={r['h']}s, m={r['m']}s")

        results = load_single_result(result_file)
        if results:
            render_results_view(results, result_file.stem)

        st.markdown("---")


def render_classification_from_predictions(variant_dir: Path, run_ts: str):
    results_dir = variant_dir / run_ts / "classification"
    if results_dir.exists():
        all_results = load_classification_results(results_dir, mode="prediction")
        # Show summary if we have multiple windows OR multiple features
        n_configs = sum(len(res) for res in all_results.values())
        if n_configs > 1 or len(all_results) > 1:
            st.markdown("## Prediction Performance Summary")
            render_classification_summary(all_results, f"pred_{run_ts}")
            st.markdown("---")

    render_classification_mode(variant_dir, run_ts, mode="prediction")


def render_classification_summary(all_results: Dict[str, Dict[Tuple[float, float], Dict[str, Any]]], key_prefix: str):
    """DRY function to render line plots grouped by feature source."""
    if not all_results:
        return

    # Check if any feature has test results
    has_any_test = any(
        any("test_results" in v for v in res.values()) 
        for res in all_results.values()
    )
    
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        metric = st.selectbox(
            "Evaluation Metric",
            options=["best_cv_score", "balanced_accuracy", "accuracy", "f1"],
            index=0,
            key=f"metric_{key_prefix}",
            format_func=lambda x: {
                "best_cv_score": "Mean CV Score (Best)",
                "balanced_accuracy": "Full-Set Balanced Accuracy",
                "accuracy": "Full-Set Accuracy",
                "f1": "Full-Set F1 Score"
            }.get(x, x)
        )
    
    # Render a separate section for each feature source
    for feature_source, feat_results in all_results.items():
        st.markdown(f"### Feature Source: {feature_source}")
        
        # Pass a single-feature Dict to maintain plotting logic but isolate the section
        single_feat_results = {feature_source: feat_results}
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                create_line_plot_by_history(single_feat_results, metric=metric),
                use_container_width=True,
                key=f"h_plot_{feature_source}_{key_prefix}",
            )
        with col2:
            st.plotly_chart(
                create_line_plot_by_future(single_feat_results, metric=metric),
                use_container_width=True,
                key=f"m_plot_{feature_source}_{key_prefix}",
            )
        st.markdown("---")


def render_classification_from_forecasts(variant_dir: Path, run_ts: str):
    results_dir = variant_dir / run_ts / "classification"
    if results_dir.exists():
        all_results = load_classification_results(results_dir, mode="forecast")
        # Show summary if we have multiple windows OR multiple features
        n_configs = sum(len(res) for res in all_results.values())
        if n_configs > 1 or len(all_results) > 1:
            st.markdown("## Forecast Performance Summary")
            render_classification_summary(all_results, f"forecast_{run_ts}")
            st.markdown("---")

    render_classification_mode(variant_dir, run_ts, mode="forecast")


def dbs_classification_tab(project_root, results_root=None):
    st.header("DBS ON/OFF Classification")

    RESULTS_ROOT = results_root if results_root else project_root / "results"

    variants = list_variants(RESULTS_ROOT)
    if len(variants) == 0:
        st.info("No result variants found under results/.")
        return

    variant = st.selectbox("Model variant", options=variants, key="class_variant")
    variant_dir = RESULTS_ROOT / variant
    runs = list_run_timestamps(variant_dir)

    if len(runs) == 0:
        st.info("No runs found for this variant yet. Train a model first.")
        return

    run_ts = st.selectbox("Run timestamp", options=runs, key="class_run")
    cfg_path = config_for_variant(project_root, variant)

    if cfg_path is None:
        st.error(f"Config not found for variant '{variant}'.")
        return

    classification_dir = variant_dir / run_ts / "classification"

    has_flipped_results = False
    hm_dirs = []
    if classification_dir.exists():
        hm_pattern = re.compile(r"^h[\d.]+_m[\d.]+$")
        hm_dirs = [
            d
            for d in classification_dir.iterdir()
            if d.is_dir() and hm_pattern.match(d.name)
        ]
        # Flipped results tab only for variants that are explicitly flipped
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

        # Reset results if variant or run changes
        current_selection = (variant, run_ts)
        if st.session_state.get("flipped_selection") != current_selection:
            st.session_state["flipped_results"] = None
            st.session_state["flipped_selection"] = current_selection

        if st.button("Load/Refresh Flipped Results"):
            with st.spinner("Loading results..."):
                st.session_state["flipped_results"] = load_classification_results(
                    classification_dir
                )

        flipped_results = st.session_state.get("flipped_results")

        if not flipped_results:
            st.info(
                "Click 'Load/Refresh Flipped Results' to visualize the (h, m) history/forecast."
            )
        else:
            st.markdown("### Flipped Performance Summary")
            render_classification_summary(flipped_results, f"flipped_{run_ts}")

            st.markdown("---")
            st.markdown("### Detailed Flipped Analysis")

            # Collect unique h, m, and feature across all results
            all_feats = sorted(flipped_results.keys())
            h_values = sorted(set(k[0] for res in flipped_results.values() for k in res.keys()))
            m_values = sorted(set(k[1] for res in flipped_results.values() for k in res.keys()))

            col_sel1, col_sel2, col_sel3 = st.columns(3)
            with col_sel1:
                sel_feat = st.selectbox("Select Feature Source", options=all_feats, key="flipped_sel_feat")
            with col_sel2:
                sel_h = st.selectbox("Select History h (s)", options=h_values, key="flipped_sel_h")
            with col_sel3:
                sel_m = st.selectbox("Select Forecast Horizon m (s)", options=m_values, key="flipped_sel_m")

            selected_res = flipped_results[sel_feat].get((sel_h, sel_m))
            if selected_res:
                st.markdown(f"#### Results for {sel_feat} - h={sel_h}s, m={sel_m}s")
                render_results_view(selected_res, f"flipped_{sel_feat}_{sel_h}_{sel_m}")
            else:
                st.warning(f"No results found for feature '{sel_feat}' with h={sel_h}s, m={sel_m}s")
    else:
        mode_tabs = st.tabs(["From Predictions", "From Forecasts"])

        with mode_tabs[0]:
            render_classification_from_predictions(variant_dir, run_ts)

        with mode_tabs[1]:
            render_classification_from_forecasts(variant_dir, run_ts)
