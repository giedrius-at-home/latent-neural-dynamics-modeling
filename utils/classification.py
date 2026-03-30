from typing import Dict, Any, List, Tuple, Optional
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV,
    permutation_test_score,
    cross_val_score,
)
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

from sklearn.base import clone
import joblib

import mne
from mne.decoding import CSP

import pickle
import polars as pl
from pathlib import Path
from utils.polars import get_scalar_value, convert_series_to_list
from utils.config import get_config


__all__ = [
    "ChronoGroupsSplit",
    "prepare_epoched_data",
    "run_grid_search_cv",
    "run_permutation_test",
    "evaluate_on_test_set",
    "create_pipeline",
    "Pipeline",
    "StandardScaler",
    "prepare_ground_truth_eval_data",
    "run_classification_pipeline",
    "run_lda_fit_and_test_only",
    "apply_lda_permutation_to_results",
    "load_precomputed_results",
    "load_all_splits",
]


class ChronoGroupsSplit:

    def __init__(
        self,
        warn_if_blocks_ignored: bool = False,
        allow_mixed_label_groups: bool = False,
    ):
        self.warn_if_blocks_ignored = warn_if_blocks_ignored
        self.allow_mixed_label_groups = allow_mixed_label_groups

    def split(self, X, y, groups):
        y = np.asarray(y)
        groups = np.asarray(groups)
        Xidcs = np.arange(X.shape[0])

        if self.allow_mixed_label_groups:
            unique_groups = sorted(set(groups))
            n_groups = len(unique_groups)

            splits = []
            for test_group in unique_groups:
                test_mask = groups == test_group
                train_mask = ~test_mask
                splits.append((Xidcs[train_mask], Xidcs[test_mask]))

            return splits

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

        if self.allow_mixed_label_groups:
            return len(set(groups))

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


def scope_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_source: str,
    n1: Optional[int],
    nx: Optional[int],
) -> np.ndarray:
    if feature_source == "Xp_1":
        return X[:, :, :n1]

    if feature_source == "Xp_2":
        return X[:, :, n1:nx]

    # Xp_with_dbs is now handled in prepare_epoched_data to avoid data leakage
    # (DBS state is added per-epoch using actual stim value, not from labels)
    # No additional processing needed here

    return X


def _dpad_idsys_forecast_latents(id_sys: Any, m: int, y_past: np.ndarray) -> np.ndarray:
    """Multi-step latent forecast for DPAD checkpoints (DPADModel has no .forecast)."""
    block_samples = id_sys.block_samples
    ny = y_past.shape[1]

    def _pad_to_block(arr: np.ndarray) -> np.ndarray:
        remainder = arr.shape[0] % block_samples
        if remainder != 0:
            pad_len = block_samples - remainder
            return np.concatenate([arr, np.zeros((pad_len, ny))], axis=0)
        return arr

    def _stack_last(steps_list):
        out = [
            arr[-1:, :] if arr is not None and len(arr.shape) == 2 else None
            for arr in steps_list
        ]
        valid = [v for v in out if v is not None]
        return np.vstack(valid) if valid else None

    id_sys.set_steps_ahead(list(range(1, m + 1)))
    id_sys.set_multi_step_with_data_gen(True, noise_samples=0)
    preds = id_sys.predict(_pad_to_block(y_past))
    xf = _stack_last(preds[2 * m : 3 * m])
    if xf is None:
        raise ValueError("DPAD predict returned no latent trajectory for forecast window")
    return xf


def _forecast_latent_trajectory(model: Any, m: int, y_past: np.ndarray) -> np.ndarray:
    """PSID idSys exposes .forecast; DPAD pickles are raw DPADModel — use predict path."""
    forecast_fn = getattr(model, "forecast", None)
    if callable(forecast_fn):
        _zf, _yf, xf = forecast_fn(m, y_past)
        if xf is None:
            raise ValueError("forecast() returned no latent trajectory (Xf is None)")
        return np.asarray(xf)
    return _dpad_idsys_forecast_latents(model, m, y_past)


def _generate_flipped_latents(
    trial_observations: np.ndarray,
    model_on: Any,
    model_off: Any,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results = []
    epoch_len, overlap = params["epoch_len"], params["overlap"]
    forecast_samples = params["forecast_samples"]
    group_id, trial_idx = params["group_id"], params["trial_idx"]
    history_samples = params["history_samples"]

    step = int(history_samples * (1 - overlap))
    T_obs = trial_observations.shape[0]

    for start in range(0, T_obs - history_samples + 1, step):
        y_past = trial_observations[start : start + history_samples]
        x_on = _forecast_latent_trajectory(model_on, forecast_samples, y_past)
        x_off = _forecast_latent_trajectory(model_off, forecast_samples, y_past)

        for label, x_traj, dtype in [
            (1, x_on, "flipped_on"),
            (0, x_off, "flipped_off"),
        ]:
            sim_epochs = epoch_trial(np.array(x_traj), epoch_len, overlap)
            for ep in sim_epochs:
                results.append(
                    {
                        "X": ep,
                        "y": label,
                        "group": group_id,
                        "meta": {
                            "type": dtype,
                            "trial_idx": trial_idx,
                            "start": start,
                            "participant_id": params.get("participant_id"),
                            "session": params.get("session"),
                            "block": params.get("block"),
                            "trial": params.get("trial"),
                        },
                    }
                )
    return results


def prepare_epoched_data(
    trials: List[Dict[str, Any]],
    feature_source: str = "Xp",
    epoch_length_sec: float = 0.5,
    overlap: float = 0.25,
    fs: float = 80,
    mode: str = "prediction",
    forecast_horizon: Optional[float] = None,
    history_horizon: Optional[float] = None,
    model_on: Optional[Any] = None,
    model_off: Optional[Any] = None,
    model_both: Optional[Any] = None,
    target_future: bool = False,
    n1: Optional[int] = None,
    nx: Optional[int] = None,
    framework: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:

    epoch_len = int(epoch_length_sec * fs)
    forecast_samples = int(forecast_horizon * fs) if forecast_horizon is not None else 0
    history_samples = int(history_horizon * fs) if history_horizon is not None else 0

    X_all, y_all, groups_all, meta_all = [], [], [], []
    block_id_map, current_block_id = {}, 0

    for trial_set in trials:
        # Skip None trial sets
        if trial_set is None:
            continue

        observations = trial_set.get("Y", [])

        # For forecast mode with history_horizon, generate forecasts on-the-fly
        # Framework is REQUIRED for this mode
        if mode == "forecast" and history_horizon is not None:
            if framework is None:
                raise ValueError(
                    f"Framework is required for forecast mode with history_horizon={history_horizon}. "
                    f"Cannot generate forecasts on-the-fly without a loaded framework."
                )
            # Generate forecasts on-the-fly with specified history window
            latent_states = []
            for trial_idx, Y_trial in enumerate(observations):
                Y_trial = np.array(Y_trial)
                if Y_trial.ndim == 1:
                    Y_trial = Y_trial.reshape(-1, 1)
                if Y_trial.shape[0] < Y_trial.shape[1]:
                    Y_trial = Y_trial.T

                # Use history_horizon to slice past data
                if Y_trial.shape[0] < history_samples + forecast_samples:
                    latent_states.append(None)
                    continue

                Y_past = Y_trial[:history_samples]
                try:
                    # Generate forecast
                    Zf, Yf, Xf = framework.model.forecast(forecast_samples, Y_past)
                    if Xf is not None:
                        latent_states.append(Xf)
                    else:
                        latent_states.append(None)
                except Exception as e:
                    warnings.warn(
                        f"Failed to generate forecast for trial {trial_idx}: {e}"
                    )
                    latent_states.append(None)
        else:
            # Use precomputed data
            data_key = "X_future_pred" if mode == "forecast" else "Xp"
            latent_states = trial_set[data_key]

        for trial_idx, trial_latents in enumerate(latent_states):
            if trial_latents is None:
                continue
            stim = trial_set["stim"][trial_idx]
            session = trial_set["session"][trial_idx]
            block = trial_set["block"][trial_idx]

            block_key = (session, block, stim)
            if block_key not in block_id_map:
                block_id_map[block_key] = current_block_id
                current_block_id += 1
            group_id = block_id_map[block_key]

            trial_latents = np.array(trial_latents)
            if trial_latents.ndim == 1:
                trial_latents = trial_latents.reshape(-1, 1)
            if trial_latents.shape[0] < trial_latents.shape[1]:
                trial_latents = trial_latents.T

            if model_on is not None and model_off is not None and target_future:
                if model_both is None:
                    raise ValueError(
                        "model_both is required for flipped classification"
                    )

                trial_obs = np.array(observations[trial_idx])
                if trial_obs.ndim == 1:
                    trial_obs = trial_obs.reshape(-1, 1)
                if trial_obs.shape[0] < trial_obs.shape[1]:
                    trial_obs = trial_obs.T

                params = {
                    "epoch_len": epoch_len,
                    "overlap": overlap,
                    "forecast_samples": forecast_samples,
                    "history_samples": history_samples,
                    "group_id": group_id,
                    "trial_idx": trial_idx,
                    "participant_id": trial_set["participant_id"][trial_idx],
                    "session": session,
                    "block": block,
                    "trial": trial_set["trial"][trial_idx],
                }
                batch = _generate_flipped_latents(
                    trial_obs, model_on, model_off, params
                )
                for item in batch:
                    X_item = item["X"]
                    # For Xp_with_dbs, add DBS state channel to each flipped sample
                    # Use actual stim value (not label) to avoid data leakage
                    if feature_source == "Xp_with_dbs":
                        dbs_channel = np.ones((X_item.shape[0], 1)) * (
                            1 if stim == "on" else 0
                        )
                        X_item = np.concatenate([X_item, dbs_channel], axis=-1)
                    X_all.append(X_item)
                    y_all.append(item["y"])
                    groups_all.append(item["group"])
                    meta_all.append(item["meta"])
                continue

            if target_future:
                if trial_latents.shape[0] < epoch_len + forecast_samples:
                    continue
                step = int(epoch_len * (1 - overlap))
                for start in range(
                    0, trial_latents.shape[0] - epoch_len - forecast_samples + 1, step
                ):
                    X_fut = trial_latents[
                        start + epoch_len : start + epoch_len + forecast_samples
                    ]
                    for sample in X_fut:
                        sample_reshaped = sample.reshape(1, -1)
                        # For Xp_with_dbs, add DBS state channel to each sample
                        if feature_source == "Xp_with_dbs":
                            dbs_channel = np.ones((1, 1)) * (1 if stim == "on" else 0)
                            sample_reshaped = np.concatenate(
                                [sample_reshaped, dbs_channel], axis=-1
                            )
                        X_all.append(sample_reshaped)
                        y_all.append(1 if stim == "on" else 0)
                        groups_all.append(group_id)
                        meta_all.append(
                            {
                                "type": "true_future",
                                "trial_idx": trial_idx,
                                "start": start,
                            }
                        )
                continue

            # For forecast mode, ensure we only use the forecast horizon
            # (already handled if generated on-the-fly, but needed for precomputed)
            if forecast_horizon is not None and mode == "forecast":
                if trial_latents.shape[0] > forecast_samples:
                    trial_latents = trial_latents[:forecast_samples]

            if trial_latents.shape[0] < epoch_len:
                continue

            epochs = epoch_trial(trial_latents, epoch_len, overlap)
            for ep_idx, ep in enumerate(epochs):
                # For Xp_with_dbs, add DBS state channel to each epoch
                # Use actual stim value (not label) to avoid data leakage
                if feature_source == "Xp_with_dbs":
                    # Add DBS state as an additional channel (1 for "on", 0 for "off")
                    dbs_channel = np.ones((ep.shape[0], 1)) * (1 if stim == "on" else 0)
                    ep = np.concatenate([ep, dbs_channel], axis=-1)

                X_all.append(ep)
                y_all.append(1 if stim == "on" else 0)
                groups_all.append(group_id)
                meta = {
                    "participant_id": trial_set["participant_id"][trial_idx],
                    "session": session,
                    "block": block,
                    "trial": trial_set["trial"][trial_idx],
                    "epoch_idx": ep_idx,
                    "group_id": group_id,
                }
                if model_on is not None:
                    meta["type"] = "real_xp"
                meta_all.append(meta)

    if not X_all:
        return None, None, None, None

    X_arr, y_arr = np.array(X_all), np.array(y_all)
    X_arr = scope_features(X_arr, y_arr, feature_source, n1, nx)

    return X_arr, y_arr, np.array(groups_all), meta_all


def prepare_ground_truth_eval_data(
    trials: List[Dict[str, Any]],
    history_horizon: float,
    forecast_horizon: float,
    fs: float,
    window_step_sec: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:

    h_samples = int(history_horizon * fs)
    m_samples = int(forecast_horizon * fs)
    window_step = int(window_step_sec * fs)

    X_all, y_all, groups_all, meta_all = [], [], [], []

    block_id_map = {}
    current_block_id = 0

    for trial_idx, trial in enumerate(trials):
        Xp = trial["Xp"]
        T = Xp.shape[0]
        stim = trial["stim"]

        block_key = (trial["session"], trial["block"], stim)
        if block_key not in block_id_map:
            block_id_map[block_key] = current_block_id
            current_block_id += 1
        group_id = block_id_map[block_key]

        if T < h_samples + m_samples:
            continue

        for start in range(0, T - h_samples - m_samples + 1, window_step):
            X_true = Xp[start + h_samples : start + h_samples + m_samples]

            X_all.append(X_true)
            y_all.append(1 if stim == "on" else 0)
            groups_all.append(group_id)
            meta_all.append(
                {
                    "trial_idx": trial_idx,
                    "start": start,
                    "stim": stim,
                    "path": trial.get("path", ""),
                }
            )

    return np.array(X_all), np.array(y_all), np.array(groups_all), meta_all


def reorder_dims_for_mne(X: np.ndarray) -> np.ndarray:
    return np.transpose(X, (0, 2, 1))


_transpose_for_csp = reorder_dims_for_mne


def create_pipeline(fs: float = 80, feature_source: str = "Xp") -> Pipeline:
    steps = [
        ("transpose", FunctionTransformer(reorder_dims_for_mne)),
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
        ("scaler", StandardScaler()),
        ("classifier", LinearDiscriminantAnalysis()),
    ]
    return Pipeline(steps)


def run_grid_search_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    fs: float = 80,
    param_grid: Optional[Dict[str, Any]] = None,
    feature_source: str = "Xp",
    allow_mixed_label_groups: bool = False,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:

    pipeline = create_pipeline(fs=fs, feature_source=feature_source)

    chrono_cv = ChronoGroupsSplit(
        warn_if_blocks_ignored=True, allow_mixed_label_groups=allow_mixed_label_groups
    )
    splits = chrono_cv.split(X, y, groups)

    effective_n_splits = len(splits)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=splits,
        scoring="balanced_accuracy",
        n_jobs=2,
        verbose=1,
        return_train_score=True,
        error_score=0.0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true"
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
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]),
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

    if logger:
        logger.info(
            f"Running group-aware permutation test ({n_permutations} permutations)..."
        )

    obs_scores = cross_val_score(
        pipeline, X, y, cv=cv, groups=groups, scoring=scoring, n_jobs=2
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
            clone(pipeline), X, y_perm, cv=cv, groups=groups, scoring=scoring, n_jobs=1
        )
        return np.mean(p_scores)

    n_parallel = 2
    permutation_scores = np.array(
        joblib.Parallel(n_jobs=n_parallel)(
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
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "roc_auc": roc_auc_val,
        "fpr": fpr,
        "tpr": tpr,
    }


def run_lda_fit_and_test_only(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    config: Any,
    logger: Any,
    feature_source: str = "Xp",
) -> Dict[str, Any]:
    clf_name = "LDA"
    n_splits = config.classification.n_splits
    sampling_freq = config.classification.sampling_freq
    param_grid = config.classification.get("param_grid", {}).get("LDA")
    flipped = config.classification.get("flipped", False)

    best_params, best_score, results = run_grid_search_cv(
        X_train,
        y_train,
        groups_train,
        n_splits,
        sampling_freq,
        param_grid=param_grid,
        feature_source=feature_source,
        allow_mixed_label_groups=flipped,
    )
    logger.info(f"  {clf_name} Balanced CV Score: {best_score:.4f}")

    if X_test is not None and y_test is not None:
        test_res = evaluate_on_test_set(results["best_pipeline"], X_test, y_test)
        results["test_results"] = test_res
        logger.info(
            f"  {clf_name} Test Balanced Acc: {test_res['balanced_accuracy']:.4f}"
        )

    return results


def apply_lda_permutation_to_results(
    results: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    config: Any,
    logger: Any,
) -> None:
    results.pop("permutation_test", None)
    n_permutations = config.classification.get("n_permutations", 100)
    flipped = config.classification.get("flipped", False)
    chrono_cv = ChronoGroupsSplit(allow_mixed_label_groups=flipped)
    p_mean, p_scores, p_val = run_permutation_test(
        results["best_pipeline"],
        X_train,
        y_train,
        groups_train,
        chrono_cv,
        n_permutations=n_permutations,
        logger=logger,
    )
    results["permutation_test"] = {
        "score": p_mean,
        "pvalue": p_val,
        "n_permutations": n_permutations,
    }


def run_classification_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    config: Any,
    logger: Any,
    feature_source: str = "Xp",
) -> Dict[str, Any]:
    results = run_lda_fit_and_test_only(
        X_train,
        y_train,
        groups_train,
        X_test,
        y_test,
        config,
        logger,
        feature_source=feature_source,
    )
    if config.classification.get("permutation_test", False):
        apply_lda_permutation_to_results(
            results, X_train, y_train, groups_train, config, logger
        )
    return results


def load_precomputed_results(
    variant_dir: Path, run_ts: str, split: str
) -> Optional[Dict[str, Any]]:

    run_dir = variant_dir / run_ts
    pickle_path = run_dir / f"{split}_results.pkl"

    if pickle_path.exists():
        try:
            with open(pickle_path, "rb") as f:
                results = pickle.load(f)
            print(f"Loaded cached {split} results from {pickle_path}")
            return results
        except Exception as e:
            print(f"Warning: Could not load pickle cache: {e}")

    new_parquet_path = variant_dir / split / f"test_results_{run_ts}.parquet"

    if not new_parquet_path.exists():
        return None
    results_path = new_parquet_path

    try:
        df = pl.read_parquet(results_path)

        # Partitioned parquet is read in filesystem order; sort to match trainer/dataloader order
        # so trial_idx aligns with (participant_id, session, block, trial)
        sort_cols = ["participant_id", "session", "block", "trial"]
        if all(c in df.columns for c in sort_cols):
            # Cast numeric partition cols from string (Hive style) for correct sort order
            df = df.with_columns(
                pl.col("session").cast(pl.Int64, strict=False),
                pl.col("block").cast(pl.Int64, strict=False),
                pl.col("trial").cast(pl.Int64, strict=False),
            ).sort(sort_cols)

        n_trials = len(df)
        cols = df.columns

        pearson_overall = get_scalar_value(df, "pearson_overall_mean")
        if pearson_overall is None:
            pearson_overall = get_scalar_value(df, "metric_pearson_r_mean")
        if pearson_overall is None:
            pearson_overall = np.nan

        pearson_overall_z = get_scalar_value(df, "pearson_overall_mean_Z")
        if pearson_overall_z is None:
            pearson_overall_z = get_scalar_value(df, "metric_pearson_r_mean_Z")
        if pearson_overall_z is None:
            pearson_overall_z = np.nan

        results = {
            "Y": convert_series_to_list(df["Y"].to_list()),
            "Yp": convert_series_to_list(df["Yp"].to_list()),
            "Z": convert_series_to_list(df["Z"].to_list()) if "Z" in cols else None,
            "Zp": (
                convert_series_to_list(df["Zp"].to_list())
                if "Zp" in cols
                else [None] * n_trials
            ),
            "Xp": (
                convert_series_to_list(df["Xp"].to_list())
                if "Xp" in cols
                else [None] * n_trials
            ),
            "pearson_per_channel": (
                convert_series_to_list(df["pearson_per_channel"].to_list())
                if "pearson_per_channel" in cols
                else (
                    convert_series_to_list(df["pearsonr_per_channel"].to_list())
                    if "pearsonr_per_channel" in cols
                    else [[np.nan]] * n_trials
                )
            ),
            "pearson_mean": (
                df["pearson_mean"].to_list()
                if "pearson_mean" in cols
                else (
                    df["pearsonr_mean"].to_list()
                    if "pearsonr_mean" in cols
                    else [np.nan] * n_trials
                )
            ),
            "pearson_overall_mean": pearson_overall,
            "pearson_per_channel_Z": (
                convert_series_to_list(df["pearsonr_per_channel_Z"].to_list())
                if "pearsonr_per_channel_Z" in cols
                else (
                    convert_series_to_list(df["pearson_per_channel_Z"].to_list())
                    if "pearson_per_channel_Z" in cols
                    else None
                )
            ),
            "pearson_mean_Z": (
                df["pearson_mean_Z"].to_list()
                if "pearson_mean_Z" in cols
                else (
                    df["pearsonr_mean_Z"].to_list()
                    if "pearsonr_mean_Z" in cols
                    else None
                )
            ),
            "pearson_overall_mean_Z": pearson_overall_z,
            "time": (
                convert_series_to_list(df["time"].to_list())
                if "time" in cols
                else [None] * n_trials
            ),
            "time_abs": (
                convert_series_to_list(df["time_abs"].to_list())
                if "time_abs" in cols
                else [None] * n_trials
            ),
            "time_margined": (
                convert_series_to_list(df["time_margined"].to_list())
                if "time_margined" in cols
                else [None] * n_trials
            ),
            "offset": (
                df["offset"].to_list() if "offset" in cols else [None] * n_trials
            ),
            "chunk_margin": (
                df["chunk_margin"].to_list()
                if "chunk_margin" in cols
                else [None] * n_trials
            ),
            "margined_duration": (
                df["margined_duration"].to_list()
                if "margined_duration" in cols
                else [None] * n_trials
            ),
            "stim": (df["stim"].to_list() if "stim" in cols else [None] * n_trials),
            "participant_id": df["participant_id"].to_list(),
            "session": df["session"].to_list(),
            "block": df["block"].to_list(),
            "trial": df["trial"].to_list(),
        }

        forecast_cols = [
            "Y_future_true",
            "Y_future_pred",
            "Y_concat_for_plot",
            "Z_future_true",
            "Z_future_pred",
            "Z_concat_for_plot",
            "X_future_pred",
        ]
        for fc in forecast_cols:
            if fc in cols:
                results[fc] = convert_series_to_list(df[fc].to_list())

        if "input_channels" in cols:
            ic_list = df["input_channels"].to_list()
            input_channels = ic_list[0] if len(ic_list) > 0 else []
            if isinstance(input_channels, pl.Series):
                input_channels = input_channels.to_list()
        else:
            input_channels = []
            for col in cols:
                if (
                    col.startswith(("ECOG_", "LFP_"))
                    and "_epochs" not in col
                    and "_psd" not in col
                ):
                    input_channels.append(col)
            input_channels = sorted(list(set(input_channels))) if input_channels else []
        results["input_channels"] = input_channels

        if "output_channels" in cols:
            oc_list = df["output_channels"].to_list()
            output_channels = oc_list[0] if len(oc_list) > 0 else []
            if isinstance(output_channels, pl.Series):
                output_channels = output_channels.to_list()
        else:
            output_channels = []
            for col in cols:
                if col in [
                    "tracing_velocity",
                    "tracing_velocity_x",
                    "tracing_velocity_y",
                    "x",
                    "y",
                ]:
                    output_channels.append(col)
        results["output_channels"] = output_channels if output_channels else []

        return results
    except Exception as e:
        warnings.warn(f"Failed to load pre-computed results for {split}: {e}")
        return None


def load_all_splits(
    variant_dir: Path, run_ts: str
) -> Dict[str, Optional[Dict[str, Any]]]:

    splits = {}
    for split_name in ["train", "val", "test"]:
        splits[split_name] = load_precomputed_results(variant_dir, run_ts, split_name)
    return splits


def _load_framework_for_forecast(
    variant_dir: Path, run_ts: str, project_root: Path
) -> Any:
    """Load framework/model for generating forecasts on-the-fly.

    Raises:
        FileNotFoundError: If model file or setup file not found
        ValueError: If framework type is unsupported or invalid
        Exception: For any other loading errors
    """
    from utils.frameworks import PSIDFramework, DPADFramework
    import json
    from utils.logger import get_logger

    logger = get_logger()

    model_path = variant_dir / f"model_{run_ts}.pkl"
    metadata_path = variant_dir / f"model_{run_ts}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Variant: {variant_dir.name}, Run: {run_ts}"
        )

    # Load metadata to determine framework type
    framework_type = "psid"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        framework_type = metadata.get("framework_type", "psid")

    # Find config file - try both training and classification setups
    setup_paths = []
    for setup_dir in ["training/setups", "classification/setups"]:
        base = project_root / setup_dir
        if base.exists():
            matches = list(base.rglob(f"{variant_dir.name}.yaml"))
            setup_paths.extend(matches)

    if not setup_paths:
        raise FileNotFoundError(
            f"Could not find setup file for variant: {variant_dir.name}. "
            f"Searched in: {project_root / 'training' / 'setups'}, "
            f"{project_root / 'classification' / 'setups'}"
        )

    config = get_config(str(setup_paths[0]))
    logger.info(f"Using config file: {setup_paths[0]}")

    # Initialize framework (only PSID and DPAD support Xp)
    if framework_type == "psid":
        framework = PSIDFramework(config)
    elif framework_type == "dpad":
        framework = DPADFramework(config)
    else:
        raise ValueError(
            f"Unknown or unsupported framework type: {framework_type}. "
            f"Only 'psid' and 'dpad' are supported for classification."
        )

    # Load model
    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)

    framework.model = framework._initialize_model()
    framework.model.idSys = model_obj
    if hasattr(model_obj, "restoreModels"):
        model_obj.restoreModels()
        logger.info("DPAD model restored successfully")

    logger.info(f"Framework loaded successfully: {framework_type}")
    return framework
