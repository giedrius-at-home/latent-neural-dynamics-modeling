import streamlit as st
import numpy as np
import pickle
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any, List

from training.components.tester import Tester
from utils.polars import get_scalar_value, convert_series_to_list


def list_variants(results_root: Path) -> List[str]:
    if not results_root.exists():
        return []
    return sorted([p.name for p in results_root.iterdir() if p.is_dir()])


def list_run_timestamps(variant_dir: Path) -> List[str]:
    ts = set()
    for p in variant_dir.glob("val_results_*"):
        name = p.name
        if name.startswith("val_results_"):
            ts.add(name.replace("val_results_", ""))
    for p in variant_dir.glob("model_*.pkl"):
        name = p.name
        if name.startswith("model_") and name.endswith(".pkl"):
            ts.add(name.replace("model_", "").replace(".pkl", ""))
    return sorted(list(ts))


def config_for_variant(project_root: Path, variant_name: str) -> Optional[Path]:
    cfg = project_root / "training" / "setups" / f"{variant_name}.yaml"
    return cfg if cfg.exists() else None


def check_precomputed_results(variant_dir: Path, run_ts: str) -> Dict[str, bool]:
    available = {}
    run_dir = variant_dir / run_ts

    for split in ["train", "val", "test"]:
        pickle_path = run_dir / f"{split}_results.pkl"
        parquet_path = variant_dir / f"{split}_results_{run_ts}"

        available[split] = pickle_path.exists() or parquet_path.exists()

    return available


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

    results_path = variant_dir / f"{split}_results_{run_ts}"
    if not results_path.exists():
        return None

    try:
        df = pl.read_parquet(results_path)
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
            "Xp": convert_series_to_list(df["Xp"].to_list()),
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
                convert_series_to_list(df["pearson_per_channel_Z"].to_list())
                if "pearson_per_channel_Z" in cols
                else (
                    convert_series_to_list(df["pearsonr_per_channel_Z"].to_list())
                    if "pearsonr_per_channel_Z" in cols
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
                    "tracing_speed",
                    "tracing_speed_x",
                    "tracing_speed_y",
                    "x",
                    "y",
                ]:
                    output_channels.append(col)
        results["output_channels"] = output_channels if output_channels else []

        return results
    except Exception as e:
        st.warning(f"Failed to load pre-computed results for {split}: {e}")
        return None


def compute_predictions_selective(
    config_path: str, run_timestamp: str, splits_to_compute: tuple
):
    tester = Tester.from_config_file(config_path, run_timestamp=run_timestamp)
    tester.run_predictions_selective(list(splits_to_compute))

    config_path_obj = Path(config_path)
    variant = config_path_obj.stem

    project_root = config_path_obj.parent.parent.parent
    results_dir = project_root / "results" / variant / run_timestamp

    for split_name in splits_to_compute:
        if split_name in tester.results:
            save_split_results(results_dir, split_name, tester.results[split_name])
            print(
                f"Saved {split_name} results to {results_dir / f'{split_name}_results.pkl'}"
            )

    return tester.results


def save_split_results(results_dir: Path, split_name: str, split_results: dict):
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{split_name}_results.pkl"

    with open(results_file, "wb") as f:
        pickle.dump(split_results, f, protocol=pickle.HIGHEST_PROTOCOL)


def compute_forecast_for_trial(
    config_path: str,
    run_ts: str,
    Y_trial: np.ndarray,
    Z_trial: Optional[np.ndarray],
    chunk_margin: Optional[float],
) -> Dict[str, Any]:
    tester = Tester.from_config_file(config_path, run_timestamp=run_ts)
    tester._load_model_for_run()
    f_res = tester.framework.model.validate_forecast(
        [Y_trial],
        Z_list=[Z_trial] if Z_trial is not None else None,
        margin=chunk_margin,
    )

    return {
        "m": f_res.get("m", 0),
        "margin_samples": f_res.get("margin_samples", 0),
        "Y_future_true": f_res.get("Y_future_true", [None])[0],
        "Y_future_pred": f_res.get("Y_future_pred", [None])[0],
        "Y_concat_for_plot": f_res.get("Y_concat_for_plot", [None])[0],
        "Z_future_true": f_res.get("Z_future_true", [None])[0],
        "Z_future_pred": f_res.get("Z_future_pred", [None])[0],
        "Z_concat_for_plot": f_res.get("Z_concat_for_plot", [None])[0],
        "pearson_per_channel": f_res.get("pearson_per_channel", [None])[0],
        "pearson_per_channel_Z": f_res.get("pearson_per_channel_Z", [None])[0],
    }


def get_trial_time_axis(
    split_res: Dict[str, Any], trial_idx: int, n_samples: int, t_offset: float = 0.0
) -> np.ndarray:
    meta_time = split_res.get("time", [])
    t_abs = meta_time[trial_idx] if meta_time and len(meta_time) > trial_idx else None

    if t_abs is None or (hasattr(t_abs, "__len__") and len(t_abs) != n_samples):
        md_list = split_res.get("margined_duration", [])
        dur = md_list[trial_idx] if md_list else None
        if dur is not None:
            t_abs = np.linspace(0.0, float(dur), n_samples)
        else:
            t_abs = np.arange(n_samples)
    else:
        t_abs = np.array(t_abs)

    return t_abs + t_offset


def transpose_if_needed(data: np.ndarray, expected_len: int) -> np.ndarray:
    if (
        data.ndim == 2
        and data.shape[0] != expected_len
        and data.shape[1] == expected_len
    ):
        return data.T
    return data
