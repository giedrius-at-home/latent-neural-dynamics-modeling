"""Disk cache for classification: flipped epoched, standard epoched, and LDA *fit* (grid search + test eval only).

Set CLASSIFICATION_DISK_CACHE=0 to disable all caches here.

LDA pickles never contain ``permutation_test``; when configs request permutations, those run after a cache hit
(``classification/compute.py`` + ``apply_lda_permutation_to_results``). Same-process repeats use an in-memory LDA fit cache.

Bump *Cache VERSION fields if epoched-feature or LDA pipeline semantics change.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FLIPPED_EPOCHED_CACHE_VERSION = 1
STANDARD_EPOCHED_CACHE_VERSION = 1
LDA_FIT_CACHE_VERSION = 2


def disk_cache_enabled() -> bool:
    return os.environ.get("CLASSIFICATION_DISK_CACHE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _fingerprint_split_artifacts(
    project_root: Path, variant_dir: Path, run_ts: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = project_root.resolve()
    for split in ("train", "val", "test"):
        pkl = variant_dir / run_ts / f"{split}_results.pkl"
        pq = variant_dir / split / f"test_results_{run_ts}.parquet"
        for p in (pkl, pq):
            if p.exists():
                st = p.stat()
                try:
                    rel = str(p.resolve().relative_to(root))
                except ValueError:
                    rel = str(p.resolve())
                rows.append(
                    {"path": rel, "mtime_ns": st.st_mtime_ns, "size": st.st_size}
                )
    rows.sort(key=lambda r: r["path"])
    return rows


def _fingerprint_model_ckpt(
    project_root: Path, variant: str, run_ts: str
) -> Dict[str, Any]:
    root = project_root.resolve()
    p = root / "results" / variant / f"model_{run_ts}.pkl"
    if not p.is_file():
        return {"path": str(p.resolve()), "missing": True}
    st = p.stat()
    try:
        rel = str(p.resolve().relative_to(root))
    except ValueError:
        rel = str(p.resolve())
    return {"path": rel, "mtime_ns": st.st_mtime_ns, "size": st.st_size}


def _fingerprint_model_metadata(
    project_root: Path, variant: str, run_ts: str
) -> Dict[str, Any]:
    root = project_root.resolve()
    p = root / "results" / variant / f"model_{run_ts}_metadata.json"
    if not p.is_file():
        return {"path": str(root / "results" / variant / f"model_{run_ts}_metadata.json"), "missing": True}
    st = p.stat()
    try:
        rel = str(p.resolve().relative_to(root))
    except ValueError:
        rel = str(p.resolve())
    return {"path": rel, "mtime_ns": st.st_mtime_ns, "size": st.st_size}


def key_hash(key: Dict[str, Any]) -> str:
    blob = json.dumps(key, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _root_epoched_dir(project_root: Path) -> Path:
    return project_root / "results" / "classification_cache" / "epoched"


def _flipped_epoched_dir(project_root: Path) -> Path:
    return project_root / "results" / "classification_cache" / "flipped"


def _lda_dir(project_root: Path) -> Path:
    return project_root / "results" / "classification_cache" / "lda_no_perm"


def _key_match(meta_key: Any, key: Dict[str, Any]) -> bool:
    return json.dumps(meta_key, sort_keys=True, default=str) == json.dumps(
        key, sort_keys=True, default=str
    )


# --- Flipped epoched ---


def build_flipped_epoched_key(
    project_root: Path,
    config: Any,
    feature_source: str,
    history_horizon: float,
    forecast_horizon: float,
    load_variant_dir: Path,
    model_run_ts: str,
) -> Dict[str, Any]:
    run = config.run
    root = Path(project_root)
    return {
        "v": FLIPPED_EPOCHED_CACHE_VERSION,
        "kind": "flipped_epoched",
        "dbs_on": f"{run.dbs_on.variant}:{run.dbs_on.run_ts}",
        "dbs_off": f"{run.dbs_off.variant}:{run.dbs_off.run_ts}",
        "dbs_both": f"{run.dbs_both.variant}:{run.dbs_both.run_ts}",
        "model_on": _fingerprint_model_ckpt(root, run.dbs_on.variant, run.dbs_on.run_ts),
        "model_off": _fingerprint_model_ckpt(root, run.dbs_off.variant, run.dbs_off.run_ts),
        "model_both": _fingerprint_model_ckpt(
            root, run.dbs_both.variant, run.dbs_both.run_ts
        ),
        "split_source": f"{load_variant_dir.name}:{model_run_ts}",
        "artifacts": _fingerprint_split_artifacts(
            project_root, load_variant_dir, model_run_ts
        ),
        "feature_source": feature_source,
        "h": float(history_horizon),
        "m": float(forecast_horizon),
        "epoch_length": float(config.classification.epoch_length),
        "overlap": float(config.classification.epoch_overlap),
        "fs": float(config.classification.sampling_freq),
        "n1": config.classification.get("n1"),
        "nx": config.classification.get("nx"),
    }


def flipped_epoched_paths(project_root: Path, kh: str) -> Tuple[Path, Path]:
    d = _flipped_epoched_dir(project_root)
    return d / f"{kh}.npz", d / f"{kh}.json"


def try_load_flipped_epoched(
    project_root: Path, key: Dict[str, Any], kh: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
    if not disk_cache_enabled():
        return None
    npz_path, json_path = flipped_epoched_paths(project_root, kh)
    if not npz_path.is_file() or not json_path.is_file():
        return None
    try:
        with open(json_path, "r") as f:
            meta = json.load(f)
        if meta.get("key_hash") != kh or not _key_match(meta.get("key"), key):
            return None
    except (json.JSONDecodeError, OSError):
        return None
    try:
        data = np.load(npz_path, allow_pickle=True)
        no_test = int(data.get("_no_test", np.array([0]))[0])
        if no_test:
            return data["X_train"], data["y_train"], data["groups_train"], None, None, None
        return (
            data["X_train"],
            data["y_train"],
            data["groups_train"],
            data["X_test"],
            data["y_test"],
            data["groups_test"],
        )
    except (OSError, KeyError, ValueError):
        return None


def save_flipped_epoched(
    project_root: Path,
    key: Dict[str, Any],
    kh: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    groups_test: Optional[np.ndarray],
) -> None:
    if not disk_cache_enabled():
        return
    d = _flipped_epoched_dir(project_root)
    d.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = flipped_epoched_paths(project_root, kh)
    if X_test is None or y_test is None or groups_test is None:
        np.savez_compressed(
            npz_path,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            _no_test=np.array([1], dtype=np.int8),
        )
    else:
        np.savez_compressed(
            npz_path,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            X_test=X_test,
            y_test=y_test,
            groups_test=groups_test,
            _no_test=np.array([0], dtype=np.int8),
        )
    with open(json_path, "w") as f:
        json.dump({"key": key, "key_hash": kh}, f, indent=2, default=str)


# --- Standard (non-flipped) epoched ---


def build_standard_epoched_key(
    project_root: Path,
    config: Any,
    feature_source: str,
    mode: str,
    history_horizon: Optional[float],
    forecast_horizon: Optional[float],
    load_variant_dir: Path,
    model_run_ts: str,
) -> Dict[str, Any]:
    root = Path(project_root)
    rv = config.run.variant
    rts = config.run.run_ts
    key: Dict[str, Any] = {
        "v": STANDARD_EPOCHED_CACHE_VERSION,
        "kind": "standard_epoched",
        "run_variant": rv,
        "run_ts": rts,
        "model_ckpt": _fingerprint_model_ckpt(root, rv, rts),
        "model_metadata": _fingerprint_model_metadata(root, rv, rts),
        "split_source": f"{load_variant_dir.name}:{model_run_ts}",
        "artifacts": _fingerprint_split_artifacts(
            project_root, load_variant_dir, model_run_ts
        ),
        "mode": mode,
        "feature_source": feature_source,
        "h": None if history_horizon is None else float(history_horizon),
        "m": None if forecast_horizon is None else float(forecast_horizon),
        "epoch_length": float(config.classification.epoch_length),
        "overlap": float(config.classification.epoch_overlap),
        "fs": float(config.classification.sampling_freq),
        "n1": config.classification.get("n1"),
        "nx": config.classification.get("nx"),
    }
    return key


def standard_epoched_paths(project_root: Path, kh: str) -> Tuple[Path, Path]:
    d = _root_epoched_dir(project_root)
    return d / f"{kh}.npz", d / f"{kh}.json"


def try_load_standard_epoched(
    project_root: Path, key: Dict[str, Any], kh: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
    if not disk_cache_enabled():
        return None
    npz_path, json_path = standard_epoched_paths(project_root, kh)
    if not npz_path.is_file() or not json_path.is_file():
        return None
    try:
        with open(json_path, "r") as f:
            meta = json.load(f)
        if meta.get("key_hash") != kh or not _key_match(meta.get("key"), key):
            return None
    except (json.JSONDecodeError, OSError):
        return None
    try:
        data = np.load(npz_path, allow_pickle=True)
        no_test = int(data.get("_no_test", np.array([0]))[0])
        if no_test:
            return data["X_train"], data["y_train"], data["groups_train"], None, None, None
        return (
            data["X_train"],
            data["y_train"],
            data["groups_train"],
            data["X_test"],
            data["y_test"],
            data["groups_test"],
        )
    except (OSError, KeyError, ValueError):
        return None


def save_standard_epoched(
    project_root: Path,
    key: Dict[str, Any],
    kh: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    groups_test: Optional[np.ndarray],
) -> None:
    if not disk_cache_enabled():
        return
    d = _root_epoched_dir(project_root)
    d.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = standard_epoched_paths(project_root, kh)
    if X_test is None or y_test is None or groups_test is None:
        np.savez_compressed(
            npz_path,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            _no_test=np.array([1], dtype=np.int8),
        )
    else:
        np.savez_compressed(
            npz_path,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            X_test=X_test,
            y_test=y_test,
            groups_test=groups_test,
            _no_test=np.array([0], dtype=np.int8),
        )
    with open(json_path, "w") as f:
        json.dump({"key": key, "key_hash": kh}, f, indent=2, default=str)


# --- LDA fit (grid + test only; no permutation_test in pickle) ---
# Used for both perm and non-perm runs: perm is always computed fresh when requested.


def build_lda_fit_key(
    epoched_key: Dict[str, Any],
    epoched_hash: str,
    config: Any,
    feature_source: str,
    mode: str,
) -> Dict[str, Any]:
    import sklearn

    return {
        "v": LDA_FIT_CACHE_VERSION,
        "kind": "lda_fit",
        "epoched_key_hash": epoched_hash,
        "epoched_key": epoched_key,
        "classification_mode": mode,
        "feature_source": feature_source,
        "param_grid": config.classification.get("param_grid"),
        "n_splits": config.classification.n_splits,
        "flipped": config.classification.get("flipped", False),
        "sklearn_version": sklearn.__version__,
    }


def lda_fit_paths(project_root: Path, kh: str) -> Tuple[Path, Path]:
    d = _lda_dir(project_root)
    return d / f"{kh}.pkl", d / f"{kh}.json"


def try_load_lda_fit(
    project_root: Path, key: Dict[str, Any], kh: str
) -> Optional[Dict[str, Any]]:
    if not disk_cache_enabled():
        return None
    pkl_path, json_path = lda_fit_paths(project_root, kh)
    if not pkl_path.is_file() or not json_path.is_file():
        return None
    try:
        with open(json_path, "r") as f:
            meta = json.load(f)
        if meta.get("key_hash") != kh or not _key_match(meta.get("key"), key):
            return None
    except (json.JSONDecodeError, OSError):
        return None
    try:
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            loaded.pop("permutation_test", None)
        return loaded
    except (OSError, pickle.UnpicklingError, EOFError):
        return None


def save_lda_fit(
    project_root: Path, key: Dict[str, Any], kh: str, results: Dict[str, Any]
) -> None:
    if not disk_cache_enabled():
        return
    d = _lda_dir(project_root)
    d.mkdir(parents=True, exist_ok=True)
    pkl_path, json_path = lda_fit_paths(project_root, kh)
    payload = {k: v for k, v in results.items() if k != "permutation_test"}
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    with open(json_path, "w") as f:
        json.dump({"key": key, "key_hash": kh}, f, indent=2, default=str)


# Backward-compatible names
build_lda_no_perm_key = build_lda_fit_key
try_load_lda_no_perm = try_load_lda_fit
save_lda_no_perm = save_lda_fit
