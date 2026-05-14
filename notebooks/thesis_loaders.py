"""Loading and data-extraction helpers for thesis sec2 notebooks.

No dashboard dependencies — imports only from utils.* and standard library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np

from utils.classification import load_precomputed_results

logger = logging.getLogger(__name__)

# Deduplicate "trial out of range" warnings across repeated calls in the same kernel session.
_oof_trial_warned: set[str] = set()

InputMode = Literal["neural", "behavioral"]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ThesisDataError(FileNotFoundError):
    """Missing or incomplete thesis results (strict loading; no silent fallbacks)."""


# ---------------------------------------------------------------------------
# Inlined low-level array helpers (originally in dashboard/subtabs/helpers.py)
# ---------------------------------------------------------------------------


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


def get_channel(data: np.ndarray, channel_idx: int, t_abs: np.ndarray) -> np.ndarray:
    data = transpose_if_needed(data, len(t_abs))
    n_chan = data.shape[1] if data.ndim == 2 else 1
    return data.squeeze() if n_chan == 1 else data[:, channel_idx]


# ---------------------------------------------------------------------------
# Inlined transform helpers (originally in thesis_lib/transforms.py)
# ---------------------------------------------------------------------------


def reshape_future_z_time_first(z: np.ndarray) -> np.ndarray:
    """Normalize Z_future / Y_future arrays to shape (n_time_steps, n_channels)."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("Z future must be 2D")
    r, c = z.shape
    if r < c and r <= 64 and c >= 2 * r:
        return z.T
    return z


def zscore_using_true_stats(true: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Map x into z-scored units defined by the ground-truth trial true."""
    true = np.asarray(true, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    mu = float(np.mean(true))
    sigma = float(np.std(true))
    if sigma < 1e-12:
        sigma = 1.0
    return (x - mu) / sigma


def z_true_and_preds(
    true: np.ndarray,
    psid: np.ndarray,
    dpad: np.ndarray,
    varma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score true and all predictions on the true trial's mean/std."""
    z_true = zscore_using_true_stats(true, true)

    def _safe_zscore(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float).ravel()
        if np.all(np.isnan(a)):
            return a
        return zscore_using_true_stats(true, arr)

    return (z_true, _safe_zscore(psid), _safe_zscore(dpad), _safe_zscore(varma))


def rmse_z(z_true: np.ndarray, z_pred: np.ndarray) -> float:
    z_true = np.asarray(z_true, dtype=float).reshape(-1)
    z_pred = np.asarray(z_pred, dtype=float).reshape(-1)
    n = min(len(z_true), len(z_pred))
    if n == 0:
        return float("nan")
    d = z_true[:n] - z_pred[:n]
    return float(np.sqrt(np.mean(d**2)))


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def load_split_results(
    results_root: Path,
    variant: str,
    run_timestamp: str,
    split: str,
) -> Optional[Dict[str, Any]]:
    variant_dir = results_root / variant
    return load_precomputed_results(variant_dir, run_timestamp, split)


def has_dpad_data(
    results_root: Path,
    variant: Optional[str],
    run_ts: Optional[str],
    split: str = "test",
) -> bool:
    """Auto-detect whether DPAD parquets are available for a triplet/split."""
    if not variant or not run_ts:
        return False
    pq_dir = results_root / variant / split / f"test_results_{run_ts}.parquet"
    if not pq_dir.is_dir():
        return False
    return next(pq_dir.rglob("0.parquet"), None) is not None


def load_split_results_required(
    results_root: Path,
    variant: str,
    run_timestamp: str,
    split: str,
) -> Dict[str, Any]:
    """Load split results or raise ThesisDataError with the expected path."""
    r = load_split_results(results_root, variant, run_timestamp, split)
    if r is not None:
        return r
    variant_dir = results_root / variant
    pq = variant_dir / split / f"test_results_{run_timestamp}.parquet"
    raise ThesisDataError(
        f"No results for variant={variant!r} run_ts={run_timestamp!r} split={split!r}. "
        f"Expected parquet directory: {pq}."
    )


# ---------------------------------------------------------------------------
# Channel metadata helpers
# ---------------------------------------------------------------------------


def channels_as_str_list(raw: Any) -> list[str]:
    """Normalize output_channels / input_channels metadata to a list of strings."""
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        seq = raw.tolist()
    elif isinstance(raw, (list, tuple)):
        seq = raw
    else:
        try:
            seq = list(raw)
        except TypeError:
            return []
    return [s for item in seq if item is not None and (s := str(item).strip())]


def output_channel_label(
    split_res: Optional[Dict[str, Any]],
    channel_idx: int,
    *,
    fallback: str = "",
    preserve_underscores: bool = False,
) -> str:
    """Human-readable output channel name from saved output_channels metadata."""
    if split_res is None or channel_idx < 0:
        return fallback
    raw = split_res.get("output_channels")
    if raw is None:
        return fallback
    if isinstance(raw, np.ndarray):
        seq = raw.tolist()
    elif isinstance(raw, (list, tuple)):
        seq = raw
    else:
        try:
            seq = list(raw)
        except TypeError:
            return fallback
    if channel_idx >= len(seq):
        return fallback
    item = seq[channel_idx]
    if item is None:
        return fallback
    s = str(item).strip()
    if not s:
        return fallback
    return s if preserve_underscores else s.replace("_", " ")


def resolve_neural_y_channel_idx(
    split_res: Dict[str, Any],
    neural_y_feature_name: str,
    fallback_channel_idx: int,
) -> int:
    """Match YAML neural_input string to input_channels list."""
    raw = (neural_y_feature_name or "").strip()
    if not raw:
        return int(fallback_channel_idx)
    names = channels_as_str_list(split_res.get("input_channels"))
    if not names:
        return int(fallback_channel_idx)
    for i, n in enumerate(names):
        if str(n).strip() == raw:
            return i
    key = raw.lower().replace(" ", "_")
    for i, n in enumerate(names):
        if str(n).lower().replace(" ", "_") == key:
            return i
    raise ValueError(
        f"Neural feature {raw!r} not found in input_channels (len={len(names)})"
    )


def resolve_neural_y_channel_idx_from_candidates(
    neural_y_feature_name: str,
    fallback_channel_idx: int,
    *candidates: Optional[Dict[str, Any]],
) -> int:
    """Like resolve_neural_y_channel_idx but tries multiple result dicts."""
    raw = (neural_y_feature_name or "").strip()
    if not raw:
        return int(fallback_channel_idx)
    key = raw.lower().replace(" ", "_")
    for split_res in candidates:
        if split_res is None:
            continue
        names = channels_as_str_list(split_res.get("input_channels"))
        if not names:
            continue
        for i, n in enumerate(names):
            if str(n).strip() == raw:
                return i
        for i, n in enumerate(names):
            if str(n).lower().replace(" ", "_") == key:
                return i
    n_nonempty = sum(
        1
        for c in candidates
        if c is not None and channels_as_str_list(c.get("input_channels"))
    )
    logger.warning(
        "resolve_neural_y_channel_idx_from_candidates: name %r not in input_channels "
        "from any of %d nonempty candidate(s); using fallback %d",
        raw,
        n_nonempty,
        fallback_channel_idx,
    )
    return int(fallback_channel_idx)


def split_res_with_nonempty_input_channels(
    *candidates: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """First candidate whose input_channels is non-empty."""
    for split_res in candidates:
        if split_res is None:
            continue
        if channels_as_str_list(split_res.get("input_channels")):
            return split_res
    return None


# ---------------------------------------------------------------------------
# Training-YAML fallback for neural_input channel lists
# ---------------------------------------------------------------------------
# Test-parquet column `input_channels` is empty for every 200Hz narrow-band run.
# The authoritative list lives in training/setups/{kind}/narrow_band_200Hz/both/{variant}.yaml
# under data.channels.neural_input. We memoise per variant.

_yaml_channel_cache: Dict[str, tuple[str, ...]] = {}


def _training_yaml_path(variant: str) -> Path:
    """Map variant name → training YAML path. Handles behavioral (narrow_band_200Hz)
    and laplacian (laplacian_200Hz) mode directories."""
    kind = variant.split("_", 1)[0]
    mode_dir = "laplacian_200Hz" if "_laplacian_" in variant else "narrow_band_200Hz"
    return Path("training/setups") / kind / mode_dir / "both" / f"{variant}.yaml"


_yaml_output_cache: Dict[str, tuple[str, ...]] = {}


def _read_yaml_channels(variant: str, key: str) -> tuple[str, ...]:
    """Read data.channels.<key> from the training YAML. Empty tuple if unavailable."""
    import yaml as _yaml

    path = _training_yaml_path(variant)
    if not path.exists():
        return ()
    cfg = _yaml.safe_load(path.read_text())
    raw = (cfg or {}).get("data", {}).get("channels", {}).get(key, [])
    return tuple(str(c) for c in raw)


_parquet_channel_cache: Dict[str, tuple[str, ...]] = {}


def _latest_test_parquet_channels(variant: str) -> tuple[str, ...]:
    """Last-resort fallback: pull input_channels from the latest test parquet
    on disk. Used when training YAMLs weren't rsynced back to this host but
    the results directory has populated input_channels (e.g. PDI1 DPAD/VARMA
    models trained on jacque)."""
    if variant in _parquet_channel_cache:
        return _parquet_channel_cache[variant]
    import glob
    import pyarrow.parquet as pq

    paths = sorted(
        glob.glob(
            f"results/{variant}/test/test_results_*.parquet/**/0.parquet",
            recursive=True,
        )
    )
    if not paths:
        _parquet_channel_cache[variant] = ()
        return ()
    try:
        df = pq.ParquetFile(paths[-1]).read().to_pandas()
        ic = df["input_channels"].iloc[0] if "input_channels" in df.columns else None
        chs = tuple(str(c) for c in ic) if ic is not None and len(ic) > 0 else ()
    except Exception:
        chs = ()
    _parquet_channel_cache[variant] = chs
    return chs


_train_log_channel_cache: Dict[str, tuple[str, ...]] = {}


def _train_log_neural_input(variant: str) -> tuple[str, ...]:
    """Last-resort fallback: parse neural_input from the training log's embedded
    config block. Used when (a) YAML wasn't rsynced back AND (b) test parquet
    doesn't store input_channels (e.g. VARMA PDI1 which dumps the config into
    logs/train_<ts>.md but lacks parquet input_channels column)."""
    import re
    import glob

    if variant in _train_log_channel_cache:
        return _train_log_channel_cache[variant]
    logs = sorted(glob.glob(f"results/{variant}/logs/train_*.md"))
    if not logs:
        _train_log_channel_cache[variant] = ()
        return ()
    try:
        text = Path(logs[-1]).read_text()
        # Match `neural_input:` followed by a bracketed list (possibly multi-line).
        m = re.search(r"neural_input\s*:\s*\[([^\]]+)\]", text, re.DOTALL)
        if not m:
            _train_log_channel_cache[variant] = ()
            return ()
        body = m.group(1)
        # tokens may be separated by whitespace or commas
        toks = [
            t.strip().strip(",").strip("'\"")
            for t in re.split(r"[\s,]+", body)
            if t.strip()
        ]
        chs = tuple(t for t in toks if t)
    except Exception:
        chs = ()
    _train_log_channel_cache[variant] = chs
    return chs


def _training_yaml_neural_input(variant: str) -> tuple[str, ...]:
    if variant in _yaml_channel_cache:
        return _yaml_channel_cache[variant]
    chs = _read_yaml_channels(variant, "neural_input")
    if not chs:
        # YAML missing (e.g. PDI1 DPAD/VARMA configs weren't rsynced back) →
        # read the latest test parquet's input_channels column.
        chs = _latest_test_parquet_channels(variant)
    if not chs:
        # Parquet doesn't have the column either (e.g. VARMA) → parse from log.
        chs = _train_log_neural_input(variant)
    _yaml_channel_cache[variant] = chs
    return chs


def _training_yaml_output(variant: str) -> tuple[str, ...]:
    if variant in _yaml_output_cache:
        return _yaml_output_cache[variant]
    chs = _read_yaml_channels(variant, "output")
    _yaml_output_cache[variant] = chs
    return chs


def resolve_input_channels(
    split_res: Optional[Dict[str, Any]], variant: str
) -> list[str]:
    """Input (Y) channel names, with YAML fallback when the parquet column is empty."""
    if split_res is not None:
        names = channels_as_str_list(split_res.get("input_channels"))
        if names:
            return names
    return list(_training_yaml_neural_input(variant))


def resolve_output_channels(
    split_res: Optional[Dict[str, Any]], variant: str
) -> list[str]:
    """Output (Z) channel names, with YAML fallback when the parquet column is empty.
    Used by laplacian-mode plots where test parquets don't store output_channels."""
    if split_res is not None:
        names = channels_as_str_list(split_res.get("output_channels"))
        if names:
            return names
    return list(_training_yaml_output(variant))


def neural_y_feature_label(
    split_res: Dict[str, Any],
    channel_idx: int,
    *,
    neural_y_feature_name: str = "",
) -> str:
    """Human-readable neural target name for captions / axes."""
    yaml_nm = (neural_y_feature_name or "").strip()
    ch = (
        resolve_neural_y_channel_idx(split_res, yaml_nm, channel_idx)
        if yaml_nm
        else int(channel_idx)
    )
    inn = channels_as_str_list(split_res.get("input_channels"))
    if 0 <= ch < len(inn):
        return str(inn[ch])
    if yaml_nm:
        return yaml_nm
    out = channels_as_str_list(split_res.get("output_channels"))
    if 0 <= ch < len(out):
        return str(out[ch])
    lab = output_channel_label(split_res, ch, fallback="", preserve_underscores=True)
    return lab if lab else f"neural Y column {ch}"


# ---------------------------------------------------------------------------
# Private trial-extraction helpers
# ---------------------------------------------------------------------------


def _prepare_z_array(
    z_trial: np.ndarray, split_res: Dict[str, Any], trial_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    z_arr = np.asarray(z_trial)
    n_samples = int(z_arr.shape[0]) if z_arr.ndim == 2 else len(z_arr)
    t_abs = get_trial_time_axis(split_res, trial_idx, n_samples)
    z_arr = transpose_if_needed(z_arr, len(t_abs))
    return z_arr, t_abs


def _n_z_trials(res: Optional[Dict[str, Any]]) -> int:
    if res is None:
        return 0
    z = res.get("Z")
    return len(z) if z is not None else 0


def _model_has_trial(res: Optional[Dict[str, Any]], trial_idx: int) -> bool:
    if res is None:
        return False
    z = res.get("Z")
    zp = res.get("Zp")
    if z is None or zp is None:
        return False
    if trial_idx >= len(z) or trial_idx >= len(zp):
        return False
    return z[trial_idx] is not None and zp[trial_idx] is not None


def _model_has_trial_y(res: Optional[Dict[str, Any]], idx: int) -> bool:
    if res is None:
        return False
    y = res.get("Y")
    yp = res.get("Yp")
    if y is None or yp is None:
        return False
    if idx >= len(y) or idx >= len(yp):
        return False
    return y[idx] is not None and yp[idx] is not None


# ---------------------------------------------------------------------------
# Trial series extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialZSeries:
    """Z_true and model Zp predictions for one trial and one output channel.

    Any model trace may be all-NaN when that model lacks the requested trial.
    """

    t_abs: np.ndarray
    z_true_raw: np.ndarray
    z_psid: np.ndarray
    z_dpad: np.ndarray
    z_varma: np.ndarray


def extract_trial_z_series(
    split_res_psid: Dict[str, Any],
    split_res_dpad: Optional[Dict[str, Any]],
    split_res_varma: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
    varma_trial_idx: Optional[int] = None,
) -> TrialZSeries:
    """Extract Z and Zp from three result dicts; PSID provides the reference time axis."""
    v_idx = varma_trial_idx if varma_trial_idx is not None else trial_idx
    if not _model_has_trial(split_res_psid, trial_idx):
        raise ValueError(
            f"PSID: missing Z/Zp for trial_idx={trial_idx} (n_trials={_n_z_trials(split_res_psid)}). "
            "PSID is the reference model; its trial must exist."
        )
    z_true_arr, t_abs = _prepare_z_array(
        split_res_psid["Z"][trial_idx], split_res_psid, trial_idx
    )
    true_c = get_channel(z_true_arr, channel_idx, t_abs)
    n = len(t_abs)

    def _zp_chan_or_nan(
        label: str, res: Optional[Dict[str, Any]], idx: int
    ) -> np.ndarray:
        if not _model_has_trial(res, idx):
            nt = _n_z_trials(res)
            key = f"{label}:{nt}"
            if key not in _oof_trial_warned:
                _oof_trial_warned.add(key)
                logger.warning(
                    "%s: trial index out of range (n_trials=%d) — traces set to NaN.",
                    label,
                    nt,
                )
            return np.full(n, np.nan)
        arr, t2 = _prepare_z_array(res["Zp"][idx], res, idx)
        if len(t2) != n:
            logger.warning(
                "%s: time-axis length mismatch (%d vs %d) — trace set to NaN",
                label,
                len(t2),
                n,
            )
            return np.full(n, np.nan)
        return get_channel(arr, channel_idx, t_abs)

    return TrialZSeries(
        t_abs=t_abs,
        z_true_raw=true_c,
        z_psid=_zp_chan_or_nan("PSID", split_res_psid, trial_idx),
        z_dpad=_zp_chan_or_nan("DPAD", split_res_dpad, trial_idx),
        z_varma=_zp_chan_or_nan("VARMA", split_res_varma, v_idx),
    )


def extract_trial_y_series(
    split_res_psid: Dict[str, Any],
    split_res_dpad: Optional[Dict[str, Any]],
    split_res_varma: Dict[str, Any],
    trial_idx: int,
    y_channel_idx: int,
    varma_trial_idx: Optional[int] = None,
) -> TrialZSeries:
    """Like extract_trial_z_series but uses neural targets Y / Yp."""
    v_idx = varma_trial_idx if varma_trial_idx is not None else trial_idx
    if not _model_has_trial_y(split_res_psid, trial_idx):
        raise ValueError(
            f"PSID: missing Y/Yp for trial_idx={trial_idx} "
            f"(n_trials={len(split_res_psid.get('Y') or [])}). "
            "Neural exemplars require saved Y and Yp on the test split."
        )
    y_true_arr, t_abs = _prepare_z_array(
        split_res_psid["Y"][trial_idx], split_res_psid, trial_idx
    )
    true_c = get_channel(y_true_arr, y_channel_idx, t_abs)
    n = len(t_abs)

    def _yp_chan_or_nan(
        label: str, res: Optional[Dict[str, Any]], idx: int
    ) -> np.ndarray:
        if not _model_has_trial_y(res, idx):
            ny = len((res or {}).get("Y") or [])
            key = f"{label}:y:{ny}"
            if key not in _oof_trial_warned:
                _oof_trial_warned.add(key)
                logger.warning(
                    "%s: Y/Yp trial index out of range (n_trials=%d) — traces set to NaN.",
                    label,
                    ny,
                )
            return np.full(n, np.nan)
        arr, t2 = _prepare_z_array(res["Yp"][idx], res, idx)
        if len(t2) != n:
            logger.warning(
                "%s: time-axis length mismatch (%d vs %d) — trace set to NaN",
                label,
                len(t2),
                n,
            )
            return np.full(n, np.nan)
        return get_channel(arr, y_channel_idx, t_abs)

    return TrialZSeries(
        t_abs=t_abs,
        z_true_raw=true_c,
        z_psid=_yp_chan_or_nan("PSID", split_res_psid, trial_idx),
        z_dpad=_yp_chan_or_nan("DPAD", split_res_dpad, trial_idx),
        z_varma=_yp_chan_or_nan("VARMA", split_res_varma, v_idx),
    )


# ---------------------------------------------------------------------------
# Single-model scoring helpers
# ---------------------------------------------------------------------------


def trial_rmse_z_for_model(
    split_res: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
) -> float:
    """RMSE on z-scored behavioral output for one trial."""
    if (
        split_res.get("Z") is None
        or trial_idx >= len(split_res["Z"])
        or split_res["Z"][trial_idx] is None
    ):
        raise ValueError(f"missing Z for trial_idx={trial_idx}")
    if (
        split_res.get("Zp") is None
        or trial_idx >= len(split_res["Zp"])
        or split_res["Zp"][trial_idx] is None
    ):
        raise ValueError(f"missing Zp for trial_idx={trial_idx}")
    z_true_arr, t_abs = _prepare_z_array(
        split_res["Z"][trial_idx], split_res, trial_idx
    )
    true_c = get_channel(z_true_arr, channel_idx, t_abs)
    zp_arr, t2 = _prepare_z_array(split_res["Zp"][trial_idx], split_res, trial_idx)
    if len(t2) != len(t_abs):
        raise ValueError("Time axis mismatch between Z and Zp for this trial.")
    pred_c = get_channel(zp_arr, channel_idx, t_abs)
    zt = zscore_using_true_stats(true_c, true_c)
    zp = zscore_using_true_stats(true_c, pred_c)
    return rmse_z(zt, zp)


def trial_rmse_y_for_model(
    split_res: Dict[str, Any],
    trial_idx: int,
    y_channel_idx: int,
) -> float:
    """RMSE on z-scored neural Y for one trial."""
    if (
        split_res.get("Y") is None
        or trial_idx >= len(split_res["Y"])
        or split_res["Y"][trial_idx] is None
    ):
        raise ValueError(f"missing Y for trial_idx={trial_idx}")
    if (
        split_res.get("Yp") is None
        or trial_idx >= len(split_res["Yp"])
        or split_res["Yp"][trial_idx] is None
    ):
        raise ValueError(f"missing Yp for trial_idx={trial_idx}")
    y_true_arr, t_abs = _prepare_z_array(
        split_res["Y"][trial_idx], split_res, trial_idx
    )
    true_c = get_channel(y_true_arr, y_channel_idx, t_abs)
    yp_arr, t2 = _prepare_z_array(split_res["Yp"][trial_idx], split_res, trial_idx)
    if len(t2) != len(t_abs):
        raise ValueError("Time axis mismatch between Y and Yp for this trial.")
    pred_c = get_channel(yp_arr, y_channel_idx, t_abs)
    zt = zscore_using_true_stats(true_c, true_c)
    zp = zscore_using_true_stats(true_c, pred_c)
    return rmse_z(zt, zp)
