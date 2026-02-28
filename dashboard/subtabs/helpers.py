import streamlit as st
import numpy as np
import pickle
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any, List
import re
from training.components.tester import Tester
from utils.polars import get_scalar_value, convert_series_to_list

from utils.classification import load_precomputed_results

def variant_short_name(variant: str) -> str:
    """Extract model type label from variant name, e.g. 'psid_PDI1_S2' -> 'PSID'."""
    return str(variant).split("_")[0].upper()


def find_baseline_variants(current_variant: str, all_variants: List[str]) -> List[str]:
    """Find baseline (VARMA) variants matching the same subject/session."""
    subject_match = re.search(r"(PDI\d+)_(?:S)?(\d+)", current_variant)
    baseline_search_str = (
        f"varma_{subject_match.group(1)}_S{subject_match.group(2)}"
        if subject_match
        else "varma"
    )
    return [
        v for v in all_variants if baseline_search_str in v and v != current_variant
    ]


def get_project_root(cfg_path: Path) -> Path:
    """Walk up from cfg_path to find the project root (parent of 'results/')."""
    for p in cfg_path.parents:
        if (p / "results").exists():
            return p
    return cfg_path


def find_config_path(project_root: Path, variant_name: str) -> Optional[Path]:
    """Recursively search for a variant's YAML config file under training/setups/ and classification/setups/."""
    for setup_dir in ["training/setups", "classification/setups"]:
        base = project_root / setup_dir
        if base.exists():
            matches = list(base.rglob(f"{variant_name}.yaml"))
            if matches:
                return matches[0]
    return None


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
    for p in variant_dir.glob("model_*_metadata.json"):
        name = p.name
        if name.startswith("model_") and name.endswith("_metadata.json"):
            ts.add(name.replace("model_", "").replace("_metadata.json", ""))
    ts_pattern = re.compile(r"^\d{8}_\d{6}$")
    for p in variant_dir.iterdir():
        if p.is_dir() and ts_pattern.match(p.name):
            ts.add(p.name)
    return sorted(list(ts))


def config_for_variant(project_root: Path, variant_name: str) -> Optional[Path]:
    training_setups = project_root / "training" / "setups"
    for cfg in training_setups.rglob(f"{variant_name}.yaml"):
        return cfg

    classification_setups = project_root / "classification" / "setups"
    for cfg in classification_setups.rglob(f"{variant_name}.yaml"):
        return cfg

    return None



def check_precomputed_results(variant_dir: Path, run_ts: str) -> Dict[str, bool]:
    available = {}
    run_dir = variant_dir / run_ts

    for split in ["train", "val", "test"]:
        pickle_path = run_dir / f"{split}_results.pkl"
        legacy_parquet_path = variant_dir / f"{split}_results_{run_ts}"
        new_parquet_path = variant_dir / split / f"test_results_{run_ts}.parquet"
 
        available[split] = (
            pickle_path.exists()
            or legacy_parquet_path.exists()
            or new_parquet_path.exists()
        )

    return available


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


def rescale_to_reference(
    pred: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    pred = np.asarray(pred).flatten()
    ref = np.asarray(ref).flatten()

    pred_mean = np.mean(pred)
    pred_std = np.std(pred)
    ref_mean = np.mean(ref)
    ref_std = np.std(ref)

    if pred_std < 1e-10:
        return np.full_like(pred, ref_mean)

    pred_rescaled = (pred - pred_mean) / pred_std * ref_std + ref_mean
    return pred_rescaled
