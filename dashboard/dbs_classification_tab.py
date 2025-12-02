import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pickle
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
import pandas as pd


def aggregate_session_features(
    split_res: Dict[str, Any], feature_type: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if feature_type == "Xp":
        data_list = split_res.get("Xp", [])
    elif feature_type == "Yp":
        data_list = split_res.get("Yp", [])
    else:
        xp_list = split_res.get("Xp", [])
        yp_list = split_res.get("Yp", [])
        data_list = []
        for xp, yp in zip(xp_list, yp_list):
            xp_arr = np.array(xp)
            yp_arr = np.array(yp)
            if xp_arr.ndim == yp_arr.ndim:
                data_list.append(np.concatenate([xp_arr, yp_arr], axis=1))
            else:
                data_list.append(None)

    stim_list = split_res.get("stim", [])
    session_list = split_res.get("session", [])
    participant_list = split_res.get("participant_id", [])
    block_list = split_res.get("block", [])

    if not data_list or not stim_list:
        return None, None, None

    session_features = {}
    session_labels = {}
    session_ids = []

    # Handle case where block_list might be missing or shorter
    if not block_list or len(block_list) != len(data_list):
        block_list = [0] * len(data_list)

    for idx, (data, stim, sess, part, blk) in enumerate(
        zip(data_list, stim_list, session_list, participant_list, block_list)
    ):
        if data is None or stim not in ["on", "off"]:
            continue

        # Aggregate by Participant + Session + Block
        session_key = f"{part}_{sess}_{blk}"

        data = np.array(data)

        if data.ndim == 2:
            if data.shape[0] < data.shape[1]:
                data = data.T
            features = np.concatenate(
                [
                    np.mean(data, axis=0),
                    np.std(data, axis=0),
                ]
            )
        else:
            features = data.flatten()

        if session_key not in session_features:
            session_features[session_key] = []
            session_labels[session_key] = stim
            session_ids.append(session_key)

        session_features[session_key].append(features)

    X = []
    y = []
    final_session_ids = []

    for session_key in session_ids:
        if session_key in session_features:
            session_feat = np.mean(session_features[session_key], axis=0)
            X.append(session_feat)
            y.append(1 if session_labels[session_key] == "on" else 0)
            final_session_ids.append(session_key)

    if len(X) == 0:
        return None, None, None

    return np.array(X), np.array(y), final_session_ids


def get_classifier(clf_name: str, params: Dict[str, Any]):
    if clf_name == "Logistic Regression":
        return LogisticRegression(**params, max_iter=1000, random_state=42)
    elif clf_name == "SVM":
        return SVC(**params, random_state=42, probability=True)
    elif clf_name == "Random Forest":
        return RandomForestClassifier(**params, random_state=42)
    elif clf_name == "Gradient Boosting":
        return GradientBoostingClassifier(**params, random_state=42)


def get_param_grid(clf_name: str) -> Dict[str, List]:
    if clf_name == "Logistic Regression":
        return {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        }
    elif clf_name == "SVM":
        return {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        }
    elif clf_name == "Random Forest":
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    elif clf_name == "Gradient Boosting":
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
        }


def run_grid_search(
    clf_name: str, X: np.ndarray, y: np.ndarray
) -> Tuple[Dict[str, Any], float]:
    param_grid = get_param_grid(clf_name)
    base_clf = get_classifier(clf_name, {})

    cv = StratifiedKFold(
        n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=42
    )

    grid_search = GridSearchCV(
        base_clf,
        param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=0,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    grid_search.fit(X_scaled, y)

    return grid_search.best_params_, grid_search.best_score_


def train_and_evaluate(
    clf_name: str, params: Dict[str, Any], X: np.ndarray, y: np.ndarray
) -> Dict[str, Any]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = get_classifier(clf_name, params)

    cv = StratifiedKFold(
        n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=42
    )

    y_pred_all = np.zeros(len(y))
    y_proba_all = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        y_pred_all[test_idx] = clf.predict(X_test)
        y_proba_all[test_idx] = clf.predict_proba(X_test)[:, 1]

    clf.fit(X_scaled, y)

    results = {
        "classifier": clf,
        "scaler": scaler,
        "params": params,
        "y_true": y,
        "y_pred": y_pred_all,
        "y_proba": y_proba_all,
        "accuracy": accuracy_score(y, y_pred_all),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred_all),
        "precision": precision_score(y, y_pred_all, zero_division=0),
        "recall": recall_score(y, y_pred_all, zero_division=0),
        "f1": f1_score(y, y_pred_all, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred_all),
    }

    fpr, tpr, _ = roc_curve(y, y_proba_all)
    results["roc_auc"] = auc(fpr, tpr)
    results["fpr"] = fpr
    results["tpr"] = tpr

    return results


def save_classification_results(results: Dict[str, Any], save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)


def load_classification_results(save_path: Path) -> Dict[str, Any]:
    if save_path.exists():
        with open(save_path, "rb") as f:
            return pickle.load(f)
    return None


def render_metrics(results: Dict[str, Any]):
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", f"{results['accuracy']:.4f}")
    with col2:
        st.metric("Balanced Accuracy", f"{results['balanced_accuracy']:.4f}")
    with col3:
        st.metric("Precision", f"{results['precision']:.4f}")
    with col4:
        st.metric("Recall", f"{results['recall']:.4f}")
    with col5:
        st.metric("F1 Score", f"{results['f1']:.4f}")


def render_confusion_matrix(cm: np.ndarray):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["off", "on"],
            y=["off", "on"],
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
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_roc_curve(results: Dict[str, Any]):
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
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_classifier_tab(
    clf_name: str, feature_type: str, split_res: Dict[str, Any], results_dir: Path
):
    st.markdown(f"### {clf_name}")

    X, y, session_ids = aggregate_session_features(split_res, feature_type)

    if X is None or y is None:
        st.warning("No valid session data available for classification")
        return

    st.write(
        f"**Sessions:** {len(X)} | **Features:** {X.shape[1]} | **DBS ON:** {np.sum(y)} | **DBS OFF:** {len(y) - np.sum(y)}"
    )

    cache_path = (
        results_dir
        / f"dbs_classification_{feature_type}_{clf_name.replace(' ', '_')}.pkl"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button(
            f"Find Best Params ({clf_name})", key=f"grid_{feature_type}_{clf_name}"
        ):
            with st.spinner("Running grid search..."):
                best_params, best_score = run_grid_search(clf_name, X, y)
                st.session_state[f"best_params_{feature_type}_{clf_name}"] = best_params
                st.session_state[f"best_score_{feature_type}_{clf_name}"] = best_score

    with col2:
        if st.button(
            f"Train & Evaluate ({clf_name})", key=f"train_{feature_type}_{clf_name}"
        ):
            params = st.session_state.get(
                f"manual_params_{feature_type}_{clf_name}",
                st.session_state.get(f"best_params_{feature_type}_{clf_name}", {}),
            )

            with st.spinner("Training classifier..."):
                results = train_and_evaluate(clf_name, params, X, y)
                save_classification_results(results, cache_path)
                st.session_state[f"results_{feature_type}_{clf_name}"] = results

    if f"best_params_{feature_type}_{clf_name}" in st.session_state:
        st.markdown("#### Best Parameters from Grid Search")
        st.json(st.session_state[f"best_params_{feature_type}_{clf_name}"])
        st.metric(
            "Best CV Balanced Accuracy",
            f"{st.session_state[f'best_score_{feature_type}_{clf_name}']:.4f}",
        )

    st.markdown("#### Manual Hyperparameters")

    if clf_name == "Logistic Regression":
        col1, col2, col3 = st.columns(3)
        with col1:
            C = st.selectbox(
                "C",
                [0.001, 0.01, 0.1, 1, 10, 100],
                index=3,
                key=f"C_{feature_type}_{clf_name}",
            )
        with col2:
            penalty = st.selectbox(
                "Penalty", ["l1", "l2"], key=f"penalty_{feature_type}_{clf_name}"
            )
        with col3:
            solver = st.selectbox(
                "Solver", ["liblinear", "saga"], key=f"solver_{feature_type}_{clf_name}"
            )

        manual_params = {"C": C, "penalty": penalty, "solver": solver}

    elif clf_name == "SVM":
        col1, col2, col3 = st.columns(3)
        with col1:
            C = st.selectbox(
                "C", [0.1, 1, 10, 100], index=1, key=f"C_{feature_type}_{clf_name}"
            )
        with col2:
            kernel = st.selectbox(
                "Kernel", ["linear", "rbf"], key=f"kernel_{feature_type}_{clf_name}"
            )
        with col3:
            gamma = st.selectbox(
                "Gamma",
                ["scale", "auto", 0.001, 0.01, 0.1],
                key=f"gamma_{feature_type}_{clf_name}",
            )

        manual_params = {"C": C, "kernel": kernel, "gamma": gamma}

    elif clf_name == "Random Forest":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_estimators = st.selectbox(
                "N Estimators",
                [50, 100, 200],
                index=1,
                key=f"n_est_{feature_type}_{clf_name}",
            )
        with col2:
            max_depth = st.selectbox(
                "Max Depth",
                [None, 10, 20, 30],
                key=f"max_depth_{feature_type}_{clf_name}",
            )
        with col3:
            min_samples_split = st.selectbox(
                "Min Samples Split",
                [2, 5, 10],
                key=f"min_split_{feature_type}_{clf_name}",
            )
        with col4:
            min_samples_leaf = st.selectbox(
                "Min Samples Leaf", [1, 2, 4], key=f"min_leaf_{feature_type}_{clf_name}"
            )

        manual_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        }

    elif clf_name == "Gradient Boosting":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_estimators = st.selectbox(
                "N Estimators",
                [50, 100, 200],
                index=1,
                key=f"n_est_{feature_type}_{clf_name}",
            )
        with col2:
            learning_rate = st.selectbox(
                "Learning Rate",
                [0.01, 0.1, 0.2],
                index=1,
                key=f"lr_{feature_type}_{clf_name}",
            )
        with col3:
            max_depth = st.selectbox(
                "Max Depth",
                [3, 5, 7],
                index=1,
                key=f"max_depth_{feature_type}_{clf_name}",
            )
        with col4:
            subsample = st.selectbox(
                "Subsample",
                [0.8, 1.0],
                index=1,
                key=f"subsample_{feature_type}_{clf_name}",
            )

        manual_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
        }

    st.session_state[f"manual_params_{feature_type}_{clf_name}"] = manual_params

    cached_results = load_classification_results(cache_path)
    results = st.session_state.get(f"results_{feature_type}_{clf_name}", cached_results)

    if results:
        st.markdown("---")
        st.markdown("#### Results")

        render_metrics(results)

        col1, col2 = st.columns(2)

        with col1:
            render_confusion_matrix(results["confusion_matrix"])

        with col2:
            render_roc_curve(results)


def render_feature_type_tab(
    feature_type: str, split_res: Dict[str, Any], results_dir: Path
):
    st.markdown(f"## {feature_type} Features")

    classifiers = ["Logistic Regression", "SVM", "Random Forest", "Gradient Boosting"]

    tabs = st.tabs(classifiers)

    for idx, clf_name in enumerate(classifiers):
        with tabs[idx]:
            render_classifier_tab(clf_name, feature_type, split_res, results_dir)


def dbs_classification_tab(project_root):
    st.header("DBS ON/OFF Classification")

    RESULTS_ROOT = project_root / "results"

    from dashboard.model_predictions_tab import (
        list_variants,
        list_run_timestamps,
        config_for_variant,
        load_precomputed_results,
    )

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
        st.error(
            f"Config not found for variant '{variant}'. Expected at training/setups/{variant}.yaml"
        )
        return

    split_choice = st.selectbox(
        "Data split", options=["train", "val", "test"], index=1, key="class_split"
    )

    if st.button("Load Data", key="btn_load_class_data"):
        st.session_state["classification_key"] = (str(cfg_path), run_ts, split_choice)

    class_key = st.session_state.get("classification_key")
    if class_key and class_key[0] == str(cfg_path) and class_key[1] == run_ts:
        split = class_key[2]

        with st.spinner(f"Loading {split} results..."):
            split_res = load_precomputed_results(variant_dir, run_ts, split)

        if split_res is None:
            st.error(
                f"No precomputed {split} results found. Please run predictions first in the Model Predictions tab."
            )
            return

        results_dir = variant_dir / run_ts

        feature_tabs = st.tabs(["Xp", "Yp", "Both"])

        with feature_tabs[0]:
            render_feature_type_tab("Xp", split_res, results_dir)

        with feature_tabs[1]:
            render_feature_type_tab("Yp", split_res, results_dir)

        with feature_tabs[2]:
            render_feature_type_tab("Both", split_res, results_dir)
