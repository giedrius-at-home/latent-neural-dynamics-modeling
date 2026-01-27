from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, FunctionTransformer
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
from mne.decoding import CSP


CLASSIFIERS = ["LDA"]


class ChronoGroupsSplit:

    def __init__(self, warn_if_blocks_ignored: bool = False):
        self.warn_if_blocks_ignored = warn_if_blocks_ignored

    def split(self, X, y, groups):
        y = np.asarray(y)
        groups = np.asarray(groups)

        gm = {k: list(set(groups[y == k])) for k in set(y)}

        for v in gm.values():
            v.sort()

        assert all(
            [set(s).intersection(groups[y != k]) == set() for k, s in gm.items()]
        ), "Groups are not unique in label"

        set_lens = [len(v) for v in gm.values()]
        if not all([e == set_lens[0] for e in set_lens]):
            min_set = min(set_lens)
            set_counts = {k: len(v) for k, v in gm.items()}
            not_assigned_groups = {k: v[min_set:] for k, v in gm.items()}

        Xidcs = np.arange(X.shape[0])

        test_idxs = np.asarray(list(zip(*[v for v in gm.values()])))

        splits = [
            (
                Xidcs[~np.isin(groups, test_idx_row)],
                Xidcs[np.isin(groups, test_idx_row)],
            )
            for test_idx_row in test_idxs
        ]

        return splits

    def get_n_splits(self, X=None, y=None, groups=None):
        if y is None or groups is None:
            raise ValueError("y and groups must be provided to get n_splits")
        y = np.asarray(y)
        groups = np.asarray(groups)
        gm = {k: list(set(groups[y == k])) for k in set(y)}
        return min(len(v) for v in gm.values())


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
    epoch_length_sec: float = 1.0,
    overlap: float = 0.5,
    fs: float = 60,
    mode: str = "prediction",
    forecast_horizon_sec: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    epoch_length = int(epoch_length_sec * fs)
    forecast_horizon = int(forecast_horizon_sec * fs) if forecast_horizon_sec else None

    X_all = []
    y_all = []
    groups_all = []
    meta_all = []

    block_id_map = {}
    current_block_id = 0

    for split_res in split_results:
        if mode == "forecast":
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
            session = session_list[trial_idx] if trial_idx < len(session_list) else 0
            block = block_list[trial_idx] if trial_idx < len(block_list) else 0

            block_key = (session, block, stim)
            if block_key not in block_id_map:
                block_id_map[block_key] = current_block_id
                current_block_id += 1
            group_id = block_id_map[block_key]

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
                groups_all.append(group_id)

                meta = {
                    "participant_id": (
                        participant_list[trial_idx]
                        if trial_idx < len(participant_list)
                        else None
                    ),
                    "session": session,
                    "block": block,
                    "trial": (
                        trial_list[trial_idx] if trial_idx < len(trial_list) else None
                    ),
                    "epoch_idx": epoch_idx,
                    "split_idx": len(meta_all),
                    "group_id": group_id,
                }

                if forecast_horizon is not None:
                    meta["forecast_horizon"] = forecast_horizon

                meta_all.append(meta)

    if len(X_all) == 0:
        return None, None, None, None

    X_array = np.array(X_all)
    y_array = np.array(y_all)
    groups_array = np.array(groups_all)

    return X_array, y_array, groups_array, meta_all


def get_classifier(clf_name: str, params: Optional[Dict[str, Any]] = None):
    params = params or {}
    if clf_name == "Logistic Regression":
        return LogisticRegression(**params, max_iter=5000, random_state=42)
    else:
        return LinearDiscriminantAnalysis(**params)


def get_param_grid(clf_name: str) -> Dict[str, List]:
    if clf_name == "LDA":
        clf_params = {
            "classifier__solver": ["lsqr"],
            "classifier__shrinkage": ["auto", 0.1, 0.5],  # auto uses Ledoit-Wolf
        }
    else:
        clf_params = {}

    return clf_params


def _transpose_for_csp(X):
    if X.ndim == 3:
        return np.transpose(X, (0, 2, 1))
    return X


def create_pipeline(clf_name: str, fs: float = 60) -> Pipeline:
    steps = [
        ("transpose", FunctionTransformer(_transpose_for_csp)),
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
        ("scaler", StandardScaler()),
        ("classifier", get_classifier(clf_name)),
    ]
    return Pipeline(steps)


def run_grid_search_cv(
    clf_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    fs: float = 60,
    param_grid: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    from sklearn.base import clone

    pipeline = create_pipeline(clf_name, fs=fs)
    if param_grid is None:
        param_grid = get_param_grid(clf_name)

    chrono_cv = ChronoGroupsSplit(warn_if_blocks_ignored=True)
    splits = chrono_cv.split(X, y, groups)

    effective_n_splits = len(splits)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=splits,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        error_score=0.0,
    )

    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_pipeline = grid_search.best_estimator_

    fold_results = []
    splits_for_details = chrono_cv.split(X, y, groups)
    for fold_idx, (train_idx, val_idx) in enumerate(splits_for_details):
        y_train, y_val = y[train_idx], y[val_idx]

        X_val = X[val_idx]
        y_pred_fold = best_pipeline.predict(X_val)
        fold_score = balanced_accuracy_score(y_val, y_pred_fold)

        fold_results.append(
            {
                "fold": fold_idx,
                "train_indices": (int(train_idx.min()), int(train_idx.max())),
                "val_indices": (int(val_idx.min()), int(val_idx.max())),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_on_train": int(np.sum(y_train == 1)),
                "n_off_train": int(np.sum(y_train == 0)),
                "n_on_val": int(np.sum(y_val == 1)),
                "n_off_val": int(np.sum(y_val == 0)),
                "accuracy": float(accuracy_score(y_val, y_pred_fold)),
                "balanced_accuracy": float(fold_score),
            }
        )

    y_pred = best_pipeline.predict(X)
    y_proba = (
        best_pipeline.predict_proba(X)[:, 1]
        if hasattr(best_pipeline, "predict_proba")
        else y_pred
    )

    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc_val = auc(fpr, tpr)

    cv_results_df = {
        "params": grid_search.cv_results_["params"],
        "mean_test_score": grid_search.cv_results_["mean_test_score"],
        "std_test_score": grid_search.cv_results_["std_test_score"],
        "mean_train_score": grid_search.cv_results_["mean_train_score"],
        "std_train_score": grid_search.cv_results_["std_train_score"],
    }

    results = {
        "best_params": best_params,
        "best_cv_score": best_score,
        "n_combinations_tested": len(grid_search.cv_results_["params"]),
        "n_splits": effective_n_splits,
        "cv_method": "ChronoGroupsSplit",
        "grid_search_results": cv_results_df,
        "fold_results": fold_results,
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


def run_permutation_test(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv: Any,
    n_permutations: int = 100,
    scoring: str = "balanced_accuracy",
    logger: Optional[Any] = None,
) -> Tuple[float, np.ndarray, float]:
    """
    Perform group-aware permutation testing.
    Standard sklearn permutation_test_score shuffles at the sample level,
    which is invalid for blocked/epoched data. This shuffles at the group level.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.base import clone
    import joblib

    if logger:
        logger.info(
            f"Running group-aware permutation test ({n_permutations} permutations)..."
        )

    obs_scores = cross_val_score(
        pipeline, X, y, cv=cv, groups=groups, scoring=scoring, n_jobs=-1
    )
    obs_mean_score = np.mean(obs_scores)

    unique_groups = np.unique(groups)
    group_to_label = {}
    for g in unique_groups:
        group_to_label[g] = y[groups == g][0]

    group_ids = list(group_to_label.keys())
    labels = np.array([group_to_label[g] for g in group_ids])

    def run_one_permutation(seed):
        rng = np.random.RandomState(seed)
        perm_labels = rng.permutation(labels)
        perm_map = dict(zip(group_ids, perm_labels))

        y_perm = np.array([perm_map[g] for g in groups])

        p_scores = cross_val_score(
            clone(pipeline), X, y_perm, cv=cv, groups=groups, scoring=scoring, n_jobs=-1
        )
        return np.mean(p_scores)

    permutation_scores = np.array(
        joblib.Parallel(n_jobs=-1)(
            joblib.delayed(run_one_permutation)(i) for i in range(n_permutations)
        )
    )

    pvalue = (np.sum(permutation_scores >= obs_mean_score) + 1) / (n_permutations + 1)
    if logger:
        logger.info(
            f"Permutation Test - Obs: {obs_mean_score:.4f}, p-value: {pvalue:.4f}"
        )

    return obs_mean_score, permutation_scores, pvalue


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
