from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    make_scorer,
)

import mne
from mne.decoding import SPoC

# TODO: Make these configurable parameters instead of hardcoded constants

SAMPLING_FREQ = 60

EPOCH_LENGTH_SEC = 1.0
EPOCH_OVERLAP = 0.5

CLASSIFIERS = ["Logistic Regression", "LDA"]


def epoch_trial(
    trial_data: np.ndarray, epoch_length: int, overlap: float = 0.5
) -> List[np.ndarray]:
    if trial_data.ndim == 1:
        trial_data = trial_data.reshape(-1, 1)

    n_samples, n_channels = trial_data.shape
    step = int(epoch_length * (1 - overlap))

    epochs = []
    for start in range(0, n_samples - epoch_length + 1, step):
        end = start + epoch_length
        epoch = trial_data[start:end, :]
        epochs.append(epoch)

    return epochs


def prepare_epoched_data(
    split_results: List[Dict[str, Any]],
    feature_source: str = "Xp",
    epoch_length_sec: float = EPOCH_LENGTH_SEC,
    overlap: float = EPOCH_OVERLAP,
    fs: float = SAMPLING_FREQ,
    forecast_horizon_sec: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    epoch_length = int(epoch_length_sec * fs)
    forecast_horizon = int(forecast_horizon_sec * fs) if forecast_horizon_sec else None

    X_all = []
    y_all = []
    meta_all = []

    for split_res in split_results:
        if forecast_horizon is not None:
            if feature_source == "Yp":
                data_key = "Y_future_pred"
            elif feature_source == "Xp":
                data_key = "X_future_pred"
            else:
                data_key = "Y_future_pred"
            data_list = split_res.get(data_key, [])
        else:
            if feature_source == "Xp":
                data_list = split_res.get("Xp", [])
            elif feature_source == "Yp":
                data_list = split_res.get("Yp", [])
            elif feature_source == "Y":
                data_list = split_res.get("Y", [])
            elif feature_source == "Both":
                xp_list = split_res.get("Xp", [])
                yp_list = split_res.get("Yp", [])
                data_list = []
                for xp, yp in zip(xp_list, yp_list):
                    xp_arr = np.array(xp)
                    yp_arr = np.array(yp)
                    if xp_arr.shape[0] < xp_arr.shape[1]:
                        xp_arr = xp_arr.T
                    if yp_arr.shape[0] < yp_arr.shape[1]:
                        yp_arr = yp_arr.T
                    data_list.append(np.concatenate([xp_arr, yp_arr], axis=1))
            else:
                data_list = split_res.get(feature_source, [])

        stim_list = split_res.get("stim", [])
        participant_list = split_res.get("participant_id", [])
        session_list = split_res.get("session", [])
        block_list = split_res.get("block", [])
        trial_list = split_res.get("trial", [])

        for trial_idx, trial_data in enumerate(data_list):

            stim = stim_list[trial_idx] if trial_idx < len(stim_list) else None

            trial_data = np.array(trial_data)
            if trial_data.ndim == 1:
                trial_data = trial_data.reshape(-1, 1)
            if trial_data.shape[0] < trial_data.shape[1]:
                trial_data = trial_data.T

            if forecast_horizon is not None:
                trial_data = trial_data[:forecast_horizon]
                if trial_data.shape[0] < epoch_length:
                    continue

            epochs = epoch_trial(trial_data, epoch_length, overlap)

            label = 1 if stim == "on" else 0

            for epoch_idx, epoch in enumerate(epochs):
                X_all.append(epoch)
                y_all.append(label)

                meta = {
                    "participant_id": (
                        participant_list[trial_idx]
                        if trial_idx < len(participant_list)
                        else None
                    ),
                    "session": (
                        session_list[trial_idx]
                        if trial_idx < len(session_list)
                        else None
                    ),
                    "block": (
                        block_list[trial_idx] if trial_idx < len(block_list) else None
                    ),
                    "trial": (
                        trial_list[trial_idx] if trial_idx < len(trial_list) else None
                    ),
                    "epoch_idx": epoch_idx,
                    "split_idx": len(meta_all),
                }

                if forecast_horizon is not None:
                    meta["forecast_horizon"] = forecast_horizon

                meta_all.append(meta)

    if len(X_all) == 0:
        return None, None, None

    X_array = np.array(X_all)
    y_array = np.array(y_all)

    return X_array, y_array, meta_all


def get_classifier(clf_name: str, params: Optional[Dict[str, Any]] = None):
    params = params or {}
    if clf_name == "Logistic Regression":
        return LogisticRegression(**params, max_iter=5000, random_state=42)
    else:
        return LinearDiscriminantAnalysis(**params)


def get_param_grid(clf_name: str) -> Dict[str, List]:
    spoc_params = {
        "spoc__n_components": [2, 4, 6, 8],
        "spoc__reg": ["empirical", "ledoit_wolf", "oas"],
        "spoc__log": [True, False],
    }

    if clf_name == "Logistic Regression":
        clf_params = {
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["liblinear", "saga"],
        }
    elif clf_name == "LDA":
        clf_params = {
            "classifier__solver": ["svd", "lsqr", "eigen"],
            "classifier__shrinkage": [None, "auto", 0.1, 0.5, 0.9],
        }
    else:
        clf_params = {}

    return {**spoc_params, **clf_params}


def create_pipeline(clf_name: str, fs: float = SAMPLING_FREQ) -> Pipeline:
    steps = [
        ("spoc", SPoC()),
        ("scaler", StandardScaler()),
        ("classifier", get_classifier(clf_name)),
    ]
    return Pipeline(steps)


def run_grid_search_cv(
    clf_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    fs: float = SAMPLING_FREQ,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    from sklearn.model_selection import ParameterGrid

    pipeline = create_pipeline(clf_name, fs=fs)
    param_grid = get_param_grid(clf_name)
    all_params = list(ParameterGrid(param_grid))

    if clf_name == "LDA":
        filtered_params = []
        for params in all_params:
            solver = params.get("classifier__solver")
            shrinkage = params.get("classifier__shrinkage")
            if solver == "svd" and shrinkage is not None:
                continue
            filtered_params.append({k: [v] for k, v in params.items()})
    else:
        filtered_params = [{k: [v] for k, v in params.items()} for params in all_params]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    grid_search = GridSearchCV(
        pipeline,
        filtered_params,
        cv=tscv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    cv_results_df = {
        "params": grid_search.cv_results_["params"],
        "mean_test_score": grid_search.cv_results_["mean_test_score"],
        "std_test_score": grid_search.cv_results_["std_test_score"],
        "mean_train_score": grid_search.cv_results_["mean_train_score"],
        "std_train_score": grid_search.cv_results_["std_train_score"],
    }

    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X)
    y_proba = (
        best_pipeline.predict_proba(X)[:, 1]
        if hasattr(best_pipeline, "predict_proba")
        else y_pred
    )

    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc_val = auc(fpr, tpr)

    results = {
        "best_params": best_params,
        "best_cv_score": best_score,
        "n_combinations_tested": len(filtered_params),
        "grid_search_results": cv_results_df,
        "y_true": y,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "roc_auc": roc_auc_val,
        "fpr": fpr,
        "tpr": tpr,
        "best_pipeline": best_pipeline,
    }

    return best_params, best_score, results


def evaluate_on_test_set(
    best_pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    y_pred = best_pipeline.predict(X_test)
    y_proba = (
        best_pipeline.predict_proba(X_test)[:, 1]
        if hasattr(best_pipeline, "predict_proba")
        else y_pred
    )

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)

    return {
        "y_true": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_val,
        "fpr": fpr,
        "tpr": tpr,
    }
