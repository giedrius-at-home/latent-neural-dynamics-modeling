"""Data preparation + on-disk loaders for LDA classification.

Prep (in-memory):
- ``epoch_trial`` / ``scope_features``: low-level per-trial helpers.
- ``_forecast_latent_trajectory``: unified dispatch for raw-checkpoint
  forecasts (DPAD vs PSID) used by the "flipped" counterfactual pass.
- ``_generate_flipped_latents``: runs both on- and off-condition models on the
  same observed window and emits epochs labelled by model identity (not by
  true stim). The flipped classifier evaluates the separability of *learned
  latent dynamics*, independent of DBS state leakage.
- ``prepare_epoched_data``: the umbrella builder consumed by the phase-5
  classification driver.
- ``prepare_ground_truth_eval_data``: fixed-horizon windowing for latent-only
  evaluation (no forecasting).

Loaders (from disk):
- ``load_precomputed_results`` / ``load_all_splits``: hydrate trainer/tester
  results per split (cache-first, parquet-fallback).
- ``_load_framework_for_forecast``: rebuild full framework + model for
  on-the-fly forecast generation during classification.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pickle
import warnings

import numpy as np
import polars as pl

from utils.config import get_config
from utils.polars import convert_series_to_list, get_scalar_value


# ─────────────────────────────────────────────────────────────────────────────
# Per-trial windowing helpers
# ─────────────────────────────────────────────────────────────────────────────


def epoch_trial(
    trial_data: np.ndarray, epoch_length: int, overlap: float = 0.5
) -> List[np.ndarray]:
    if trial_data.ndim == 1:
        trial_data = trial_data.reshape(-1, 1)
    n_samples, _ = trial_data.shape
    step = int(epoch_length * (1 - overlap))
    epochs = []
    for start in range(0, n_samples - epoch_length + 1, step):
        epochs.append(trial_data[start : start + epoch_length, :])
    return epochs


def scope_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_source: str,
    n1: Optional[int],
    nx: Optional[int],
) -> np.ndarray:
    """Slice latent dims by feature source. Xp_with_dbs is handled per-epoch
    in ``prepare_epoched_data`` so DBS state comes from actual stim (not labels)."""
    if feature_source == "Xp_1":
        return X[:, :, :n1]
    if feature_source == "Xp_2":
        return X[:, :, n1:nx]
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Forecast helpers (raw idSys → latent trajectory)
# ─────────────────────────────────────────────────────────────────────────────

# Per-process memoisation for DPAD forecast setup. ``set_steps_ahead(1..m)`` and
# ``set_multi_step_with_data_gen`` each rebuild m output heads on the TF model
# (O(seconds) per call). Cache keyed by Python ``id`` of the idSys instance.
_DPAD_FORECAST_SETUP_CACHE: Dict[int, int] = {}


def _ensure_dpad_forecast_setup(id_sys: Any, m: int) -> None:
    key = id(id_sys)
    if _DPAD_FORECAST_SETUP_CACHE.get(key) != m:
        id_sys.set_steps_ahead(list(range(1, m + 1)))
        id_sys.set_multi_step_with_data_gen(True, noise_samples=0)
        _DPAD_FORECAST_SETUP_CACHE[key] = m


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

    _ensure_dpad_forecast_setup(id_sys, m)
    preds = id_sys.predict(_pad_to_block(y_past))
    xf = _stack_last(preds[2 * m : 3 * m])
    if xf is None:
        raise ValueError(
            "DPAD predict returned no latent trajectory for forecast window"
        )
    return xf


def _forecast_latent_trajectory(model: Any, m: int, y_past: np.ndarray) -> np.ndarray:
    """Framework-agnostic forecast dispatch.

    DPAD checkpoints carry a ``block_samples`` attribute; PSID LSSMs don't. For
    PSID we route through ``PSIDWrapper.from_idsys`` so we inherit the same
    forecast machinery used during training.
    """
    if hasattr(model, "block_samples"):
        return _dpad_idsys_forecast_latents(model, m, y_past)
    from utils.frameworks import PSIDWrapper

    _zf, _yf, xf = PSIDWrapper.from_idsys(model).forecast(m, y_past)
    if xf is None:
        raise ValueError(
            "PSIDWrapper.forecast returned no latent trajectory (Xf is None)"
        )
    return np.asarray(xf)


def _generate_flipped_latents(
    trial_observations: np.ndarray,
    model_on: Any,
    model_off: Any,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run both on- and off-models on each sliding window; emit epochs labelled
    by model identity. Forms the "flipped" classification pool.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Epoched-data builders (public)
# ─────────────────────────────────────────────────────────────────────────────


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
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[List[Dict[str, Any]]],
]:

    epoch_len = int(epoch_length_sec * fs)
    forecast_samples = int(forecast_horizon * fs) if forecast_horizon is not None else 0
    history_samples = int(history_horizon * fs) if history_horizon is not None else 0

    X_all, y_all, groups_all, meta_all = [], [], [], []
    block_id_map: Dict[Any, int] = {}
    current_block_id = 0

    for trial_set in trials:
        if trial_set is None:
            continue

        observations = trial_set.get("Y", [])

        # Forecast-mode requires a framework to produce latent trajectories live.
        # Exception: flipped pass supplies model_on/model_off instead; fall through to else.
        if mode == "forecast" and history_horizon is not None and model_on is None:
            if framework is None:
                raise ValueError(
                    f"Framework is required for forecast mode with "
                    f"history_horizon={history_horizon}. "
                    f"Cannot generate forecasts on-the-fly without a loaded framework."
                )
            latent_states = []
            for trial_idx, Y_trial in enumerate(observations):
                Y_trial = np.array(Y_trial)
                if Y_trial.ndim == 1:
                    Y_trial = Y_trial.reshape(-1, 1)
                if Y_trial.shape[0] < Y_trial.shape[1]:
                    Y_trial = Y_trial.T

                if Y_trial.shape[0] < history_samples + forecast_samples:
                    latent_states.append(None)
                    continue

                Y_past = Y_trial[:history_samples]
                try:
                    _Zf, _Yf, Xf = framework.model.forecast(forecast_samples, Y_past)
                    latent_states.append(Xf if Xf is not None else None)
                except Exception as e:
                    warnings.warn(
                        f"Failed to generate forecast for trial {trial_idx}: {e}"
                    )
                    latent_states.append(None)
        else:
            # Flipped forecast uses Xp as placeholder; _generate_flipped_latents overwrites per-trial.
            if mode == "forecast" and model_on is None:
                data_key = "X_future_pred"
            else:
                data_key = "Xp"
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

            # Flipped (counterfactual) pass: two models, two labels, same obs.
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

            # True-future pass: per-sample labels from ground-truth latent tail.
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

            # Forecast-mode sanity clip (precomputed path).
            if forecast_horizon is not None and mode == "forecast":
                if trial_latents.shape[0] > forecast_samples:
                    trial_latents = trial_latents[:forecast_samples]

            if trial_latents.shape[0] < epoch_len:
                continue

            epochs = epoch_trial(trial_latents, epoch_len, overlap)
            for ep_idx, ep in enumerate(epochs):
                if feature_source == "Xp_with_dbs":
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
    """Extract ``m``-sample windows from the *true* latent trajectory, skipping
    the first ``history`` samples of each trial. No forecasting — pure slicing.
    """
    h_samples = int(history_horizon * fs)
    m_samples = int(forecast_horizon * fs)
    window_step = int(window_step_sec * fs)

    X_all, y_all, groups_all, meta_all = [], [], [], []
    block_id_map: Dict[Any, int] = {}
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


# ─────────────────────────────────────────────────────────────────────────────
# On-disk loaders
# ─────────────────────────────────────────────────────────────────────────────


def load_precomputed_results(
    variant_dir: Path, run_ts: str, split: str
) -> Optional[Dict[str, Any]]:
    """Try cached results blob first; fall back to parquet partition."""
    run_dir = variant_dir / run_ts
    cache_path = run_dir / f"{split}_results.pkl"

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                results = pickle.load(f)
            print(f"Loaded cached {split} results from {cache_path}")
            return results
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")

    parquet_path = variant_dir / split / f"test_results_{run_ts}.parquet"
    if not parquet_path.exists():
        return None

    try:
        df = pl.read_parquet(parquet_path)

        sort_cols = ["participant_id", "session", "block", "trial"]
        if all(c in df.columns for c in sort_cols):
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

        results: Dict[str, Any] = {
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
            "stim": df["stim"].to_list() if "stim" in cols else [None] * n_trials,
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
            input_channels = sorted(
                {
                    col
                    for col in cols
                    if col.startswith(("ECOG_", "LFP_"))
                    and "_epochs" not in col
                    and "_psd" not in col
                }
            )
        results["input_channels"] = input_channels

        if "output_channels" in cols:
            oc_list = df["output_channels"].to_list()
            output_channels = oc_list[0] if len(oc_list) > 0 else []
            if isinstance(output_channels, pl.Series):
                output_channels = output_channels.to_list()
        else:
            output_channels = [
                col
                for col in cols
                if col
                in (
                    "tracing_velocity",
                    "tracing_velocity_x",
                    "tracing_velocity_y",
                    "x",
                    "y",
                )
            ]
        results["output_channels"] = output_channels if output_channels else []

        return results
    except Exception as e:
        warnings.warn(f"Failed to load pre-computed results for {split}: {e}")
        return None


def load_all_splits(
    variant_dir: Path, run_ts: str
) -> Dict[str, Optional[Dict[str, Any]]]:
    return {
        split: load_precomputed_results(variant_dir, run_ts, split)
        for split in ("train", "val", "test")
    }


def _load_framework_for_forecast(
    variant_dir: Path, run_ts: str, project_root: Path, config: Optional[Any] = None
) -> Any:
    """Rebuild framework + load trained model for on-the-fly forecast generation.

    ``config`` may be passed directly (e.g. from the pipeline) to skip the
    YAML search. When omitted the function falls back to searching
    ``training/setups`` and ``classification/setups`` for a matching YAML.
    """
    from utils.frameworks import DPADFramework, PSIDFramework
    from utils.logger import get_logger

    logger = get_logger()

    model_path = variant_dir / f"model_{run_ts}.pkl"
    metadata_path = variant_dir / f"model_{run_ts}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Variant: {variant_dir.name}, Run: {run_ts}"
        )

    framework_type = "psid"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        framework_type = metadata.get("framework_type", "psid")

    if config is None:
        import re

        setup_paths = []
        for setup_dir in ("training/setups", "classification/setups"):
            base = project_root / setup_dir
            if base.exists():
                setup_paths.extend(base.rglob(f"{variant_dir.name}.yaml"))
                if not setup_paths:
                    exp_name = re.sub(r"_dbs_(both|on|off)$", "", variant_dir.name)
                    setup_paths.extend(base.rglob(f"{exp_name}.yaml"))
        if not setup_paths:
            raise FileNotFoundError(
                f"Could not find setup file for variant: {variant_dir.name}. "
                f"Searched under training/setups and classification/setups."
            )
        config = get_config(str(setup_paths[0]))
        logger.info(f"Using config file: {setup_paths[0]}")

    if framework_type == "psid":
        framework = PSIDFramework(config)
    elif framework_type == "dpad":
        framework = DPADFramework(config)
    else:
        raise ValueError(
            f"Unknown framework type: {framework_type}. Only 'psid' and 'dpad' supported."
        )

    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)

    framework.model = framework._initialize_model()
    framework.model.idSys = model_obj
    if hasattr(model_obj, "restoreModels"):
        model_obj.restoreModels()
        logger.info("DPAD model restored successfully")

    logger.info(f"Framework loaded successfully: {framework_type}")
    return framework
