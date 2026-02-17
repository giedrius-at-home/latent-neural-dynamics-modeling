import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import pickle
import pandas as pd

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
    create_summary_table,
)


def load_all_splits(
    variant_dir: Path, run_ts: str
) -> Dict[str, Optional[Dict[str, Any]]]:
    from dashboard.subtabs import load_precomputed_results

    splits = {}
    for split_name in ["train", "val", "test"]:
        splits[split_name] = load_precomputed_results(variant_dir, run_ts, split_name)
    return splits


def save_classification_results(results: Dict[str, Any], save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)


def load_single_result(save_path: Path) -> Optional[Dict[str, Any]]:
    if save_path.exists():
        with open(save_path, "rb") as f:
            return pickle.load(f)
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
            mode="lines",
            name=f"ROC (AUC = {results['roc_auc']:.4f})",
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
        xaxis_title="FP Rate",
        yaxis_title="TP Rate",
        height=350,
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family),
        margin=dict(l=50, r=20, t=10, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"roc_{key}")


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

    if results:
        if "best_params" in results:
            st.markdown("#### Best Hyperparameters")
            st.json(results["best_params"])
            cols = st.columns(2)
        st.markdown("#### Cross-Validation Results")
        render_metrics_row(results, "CV ")

        # if "fold_results" in results:
        #     with st.expander("Fold Details", expanded=False):
        #         render_fold_results(results["fold_results"], key_base)

        col1, col2 = st.columns(2)
        with col1:
            render_confusion_matrix(results["confusion_matrix"], f"cv_{key_base}")
        with col2:
            render_roc_curve(results, f"cv_{key_base}")

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
                render_confusion_matrix(
                    test_res["confusion_matrix"], f"test_{key_base}"
                )
            with col2:
                render_roc_curve(test_res, f"test_{key_base}")


def render_classification_mode(
    variant_dir: Path,
    run_ts: str,
    mode: str,
):

    results_dir = variant_dir / run_ts / "classification"

    if not results_dir.exists():
        st.warning(f"No classification directory found at {results_dir.relative_to(variant_dir.parent.parent)}. Run the classification script first.")
        return

    # Look for results in main dir AND subdirs
    pattern = f"*_{mode}.pkl"
    result_files = list(results_dir.rglob(pattern))

    if not result_files:
        st.info(f"No {mode} classification results found. Run the classification script to generate results.")
        return

    # Group by (h, m) if applicable
    import re
    hm_pattern = re.compile(r"h([\d.]+)_m([\d.]+)")
    
    config_map = {}
    for f in result_files:
        match = hm_pattern.search(f.parent.name)
        if match:
            h, m = float(match.group(1)), float(match.group(2))
            config_map[f"h={h}s, m={m}s"] = f
        else:
            config_map["Standard (1.0s)"] = f
    
    if len(config_map) > 1:
        selected_cfg = st.selectbox(f"Window Configuration (h, m) - {mode}", options=list(config_map.keys()), key=f"sel_{mode}_{run_ts}")
        display_files = [config_map[selected_cfg]]
    else:
        display_files = result_files

    for result_file in sorted(display_files):
        filename = result_file.stem
        parts = filename.split("_")

        clf_name = parts[0]  # LDA
        feature_source = parts[1]  # Xp, Yp, etc

        epoch_info = [p for p in parts if p.startswith("epoch")]
        overlap_info = [p for p in parts if p.startswith("overlap")]

        epoch_str = epoch_info[0] if epoch_info else "unknown"
        overlap_str = overlap_info[0] if overlap_info else "unknown"

        display_name = f"{clf_name} - {feature_source} features"
        params_str = f"{epoch_str}, {overlap_str}"

        st.markdown(f"### {display_name}")
        st.caption(f"Parameters: {params_str}")

        results = load_single_result(result_file)

        if results:
            if "best_params" in results:
                st.markdown("#### Best Hyperparameters")
                st.json(results["best_params"])

            st.markdown("#### Cross-Validation Results")
            render_metrics_row(results, "CV ")

            # if "fold_results" in results:
            #     with st.expander("Fold Details", expanded=False):
            #         render_fold_results(results["fold_results"], filename)

            col1, col2 = st.columns(2)
            with col1:
                render_confusion_matrix(results["confusion_matrix"], f"cv_{filename}")
            with col2:
                render_roc_curve(results, f"cv_{filename}")

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
                    render_confusion_matrix(
                        test_res["confusion_matrix"], f"test_{filename}"
                    )
                with col2:
                    render_roc_curve(test_res, f"test_{filename}")

        st.markdown("---")


def render_classification_from_predictions(variant_dir: Path, run_ts: str):
    render_classification_mode(variant_dir, run_ts, mode="prediction")


def render_classification_from_forecasts(variant_dir: Path, run_ts: str):
    results_dir = variant_dir / run_ts / "classification"
    if results_dir.exists():
        all_mode_results = load_classification_results(results_dir, mode="forecast")
        if len(all_mode_results) > 1:
            st.markdown("## Forecast Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_line_plot_by_history(all_mode_results, metric="balanced_accuracy"), use_container_width=True, key="h_fore")
            with col2:
                st.plotly_chart(create_line_plot_by_future(all_mode_results, metric="balanced_accuracy"), use_container_width=True, key="m_fore")
            st.markdown("---")
            
    render_classification_mode(variant_dir, run_ts, mode="forecast")


def dbs_classification_tab(project_root, results_root=None):
    st.header("DBS ON/OFF Classification")

    RESULTS_ROOT = results_root if results_root else project_root / "results"

    from dashboard.subtabs import list_variants, list_run_timestamps, config_for_variant

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
        import re
        hm_pattern = re.compile(r"^h[\d.]+_m[\d.]+$")
        hm_dirs = [d for d in classification_dir.iterdir() if d.is_dir() and hm_pattern.match(d.name)]
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
        st.markdown("## Classification Results on Different History and Forecast Windows")
        
        # Reset results if variant or run changes
        current_selection = (variant, run_ts)
        if st.session_state.get("flipped_selection") != current_selection:
            st.session_state["flipped_results"] = None
            st.session_state["flipped_selection"] = current_selection

        if st.button("Load/Refresh Flipped Results"):
            with st.spinner("Loading results..."):
                st.session_state["flipped_results"] = load_classification_results(classification_dir)
        
        flipped_results = st.session_state.get("flipped_results")
        
        if not flipped_results:
            st.info("Click 'Load/Refresh Flipped Results' to visualize the (h, m) history/forecast.")
        else:
            
            valid_test_results = {k: v for k, v in flipped_results.items() if "test_results" in v}
            
            if not valid_test_results:
                st.warning("No valid test results found. Note: m must be >= epoch_length (1.0s) to generate test samples.")
                plot_results = flipped_results
                plot_metric = "best_cv_score"
            else:
                plot_results = valid_test_results
                plot_metric = "balanced_accuracy"

            st.markdown("### Flipped Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_line_plot_by_history(plot_results, metric=plot_metric), use_container_width=True, key="sweep_h_flipped")
            with col2:
                st.plotly_chart(create_line_plot_by_future(plot_results, metric=plot_metric), use_container_width=True, key="sweep_m_flipped")
            
            st.markdown("---")
            st.markdown("### Detailed Flipped Analysis")
            
            h_values = sorted(set(k[0] for k in flipped_results.keys()))
            m_values = sorted(set(k[1] for k in flipped_results.keys()))
            
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                sel_h = st.selectbox("Select History h (s)", options=h_values, key="flipped_sel_h")
            with col_sel2:
                sel_m = st.selectbox("Select Forecast Horizon m (s)", options=m_values, key="flipped_sel_m")
            
            selected_res = flipped_results.get((sel_h, sel_m))
            if selected_res:
                st.markdown(f"#### Results for h={sel_h}s, m={sel_m}s")
                
                st.markdown("#### Cross-Validation Results")
                render_metrics_row(selected_res, "CV ")

                col1, col2 = st.columns(2)
                with col1:
                    render_confusion_matrix(selected_res["confusion_matrix"], f"cv_flipped_{sel_h}_{sel_m}")
                with col2:
                    render_roc_curve(selected_res, f"cv_flipped_{sel_h}_{sel_m}")

                if "test_results" in selected_res:
                    st.markdown("---")
                    st.markdown("#### Test Set Results (Held Out)")
                    test_res = selected_res["test_results"]
                    render_metrics_row(test_res, "Test ")

                    col1, col2 = st.columns(2)
                    with col1:
                        render_confusion_matrix(test_res["confusion_matrix"], f"test_flipped_{sel_h}_{sel_m}")
                    with col2:
                        render_roc_curve(test_res, f"test_flipped_{sel_h}_{sel_m}")
            else:
                st.warning(f"No results found for h={sel_h}s, m={sel_m}s")
    else:
        mode_tabs = st.tabs(["From Predictions", "From Forecasts"])

        with mode_tabs[0]:
            render_classification_from_predictions(variant_dir, run_ts)

        with mode_tabs[1]:
            render_classification_from_forecasts(variant_dir, run_ts)
