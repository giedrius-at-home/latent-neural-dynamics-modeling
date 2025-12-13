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
    SAMPLING_FREQ,
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


def load_classification_results(save_path: Path) -> Optional[Dict[str, Any]]:
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
            z=cm,
            x=["OFF", "ON"],
            y=["OFF", "ON"],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
            showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        height=350,
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
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="red", dash="dash", width=1),
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=350,
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
    )
    st.plotly_chart(fig, use_container_width=True, key=f"fold_perf_{key}")


def render_classifier_section(
    clf_name: str,
    feature_source: str,
    epoch_params: str,
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
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
    if forecast_horizon_sec:
        key_base += f"_fh{forecast_horizon_sec}s".replace(".", "p")

    cache_path = results_dir / f"{key_base}.pkl"

    st.markdown(f"### {clf_name}")

    cached = load_classification_results(cache_path)
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
            with st.spinner("Running grid search with TimeSeriesSplit CV..."):
                best_params, best_score, cv_results = run_grid_search_cv(
                    clf_name, X_trainval, y_trainval, n_splits
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
            cols[0].metric("Best CV Score", f"{results.get('best_cv_score', 0):.4f}")
            cols[1].metric(
                "Combinations Tested", results.get("n_combinations_tested", 0)
            )

        st.markdown("#### Cross-Validation Results (Train+Val)")
        render_metrics_row(results, "CV ")

        if "fold_results" in results:
            with st.expander("Fold Details", expanded=False):
                render_fold_results(results["fold_results"], key_base)

        col1, col2 = st.columns(2)
        with col1:
            render_confusion_matrix(results["confusion_matrix"], f"cv_{key_base}")
        with col2:
            render_roc_curve(results, f"cv_{key_base}")

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
    data_prep_fn: Callable,
    extra_params: dict = None,
):
    st.subheader("Configuration")

    extra_params = extra_params or {}
    num_cols = 3 + len(extra_params)
    cols = st.columns(num_cols)

    with cols[0]:
        feature_source_options = (
            ["Yp", "Xp"] if mode == "forecast" else ["Xp", "Yp", "Y", "Both"]
        )
        feature_source = st.selectbox(
            "Feature Source",
            options=feature_source_options,
            index=0,
            key=f"{mode}_feat_source",
        )

    with cols[1]:
        epoch_length_sec = st.slider(
            "Epoch Length (sec)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            key=f"{mode}_epoch_length",
            help="Length of time windows to extract from each trial",
        )

    with cols[2]:
        epoch_overlap = st.slider(
            "Epoch Overlap",
            min_value=0.0,
            max_value=0.75,
            value=0.5,
            step=0.25,
            key=f"{mode}_epoch_overlap",
            help="Overlap between consecutive epochs (0=no overlap, 0.5=50% overlap)",
        )

    n_splits = st.slider(
        "CV Folds", min_value=2, max_value=10, value=5, key=f"{mode}_n_splits"
    )

    forecast_horizon_sec = None
    extra_param_values = {}
    for idx, (param_name, param_config) in enumerate(extra_params.items()):
        if param_config["type"] == "slider":
            extra_param_values[param_name] = st.slider(
                param_config["label"],
                min_value=param_config["min"],
                max_value=param_config["max"],
                value=param_config["default"],
                step=param_config.get("step", 1),
                key=f"{mode}_{param_name}",
            )
            if param_name == "forecast_horizon_sec":
                forecast_horizon_sec = extra_param_values[param_name]

    data_key_tuple = (
        variant_dir,
        run_ts,
        feature_source,
        epoch_length_sec,
        epoch_overlap,
        mode,
        tuple(extra_param_values.items()),
    )

    if st.button("Load Data", key=f"btn_load_{mode}"):
        st.session_state[f"{mode}_data_key"] = data_key_tuple

    if not st.session_state.get(f"{mode}_data_key"):
        st.info(
            f"Click 'Load Data' to prepare features from {'forecasts' if mode == 'forecast' else 'predictions'}."
        )
        return

    with st.spinner("Loading all splits..."):
        all_splits = load_all_splits(variant_dir, run_ts)

    train_res = all_splits.get("train")
    val_res = all_splits.get("val")
    test_res = all_splits.get("test")

    trainval_list = [r for r in [train_res, val_res] if r is not None]
    test_list = [test_res] if test_res is not None else []

    if not trainval_list:
        st.error("No train or val results found. Run predictions first.")
        return

    with st.spinner("Extracting epoched features..."):
        prep_kwargs = {
            "feature_source": feature_source,
            "epoch_length_sec": epoch_length_sec,
            "overlap": epoch_overlap,
        }
        prep_kwargs.update(extra_param_values)

        X_trainval, y_trainval, meta_trainval = data_prep_fn(
            trainval_list, **prep_kwargs
        )
        X_test, y_test, meta_test = (
            data_prep_fn(test_list, **prep_kwargs) if test_list else (None, None, None)
        )

    if X_trainval is None or len(X_trainval) == 0:
        st.error("No valid epochs found for classification.")
        return

    st.markdown("---")
    st.subheader("Data Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Train + Val (for CV)**")
        st.write(f"Epochs: {len(X_trainval)} | Features: {X_trainval.shape[1]}")
        st.write(
            f"DBS ON: {np.sum(y_trainval == 1)} | DBS OFF: {np.sum(y_trainval == 0)}"
        )

    with col2:
        st.markdown("**Test (Held Out)**")
        if X_test is not None and len(X_test) > 0:
            st.write(f"Epochs: {len(X_test)} | Features: {X_test.shape[1]}")
            st.write(f"DBS ON: {np.sum(y_test == 1)} | DBS OFF: {np.sum(y_test == 0)}")
        else:
            st.write("No test data available")

    st.markdown("---")
    results_dir = variant_dir / run_ts / "classification"

    epoch_params_str = f"epoch{epoch_length_sec}s_overlap{epoch_overlap}"

    tabs = st.tabs(CLASSIFIERS)
    for idx, clf_name in enumerate(CLASSIFIERS):
        with tabs[idx]:
            render_classifier_section(
                clf_name,
                feature_source,
                epoch_params_str,
                X_trainval,
                y_trainval,
                X_test,
                y_test,
                results_dir,
                n_splits,
                mode=mode,
                forecast_horizon_sec=forecast_horizon_sec,
            )


def render_classification_from_predictions(variant_dir: Path, run_ts: str):
    render_classification_mode(
        variant_dir,
        run_ts,
        mode="prediction",
        data_prep_fn=prepare_epoched_data,
    )


def render_classification_from_forecasts(variant_dir: Path, run_ts: str):
    render_classification_mode(
        variant_dir,
        run_ts,
        mode="forecast",
        data_prep_fn=prepare_epoched_data,
        extra_params={
            "forecast_horizon_sec": {
                "type": "slider",
                "label": "Forecast Horizon (sec)",
                "min": 0.5,
                "max": 10.0,
                "default": 2.5,
                "step": 0.5,
            },
        },
    )


def dbs_classification_tab(project_root):
    st.header("DBS ON/OFF Classification")

    RESULTS_ROOT = project_root / "results"

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
    if classification_dir.exists():
        num_results = len(list(classification_dir.glob("*.pkl")))
        if num_results > 0:
            st.success(f"Found {num_results} precomputed classification result(s)")
        else:
            st.warning(
                "No precomputed results found. Run the classification script first."
            )
    else:
        st.warning(
            "No classification directory found. Run the classification script first."
        )

    st.markdown("---")
    mode_tabs = st.tabs(["From Predictions", "From Forecasts"])

    with mode_tabs[0]:
        render_classification_from_predictions(variant_dir, run_ts)

    with mode_tabs[1]:
        render_classification_from_forecasts(variant_dir, run_ts)
