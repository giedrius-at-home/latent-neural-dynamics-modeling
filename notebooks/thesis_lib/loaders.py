from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence

import logging

import numpy as np

from thesis_lib._helpers import get_channel, get_trial_time_axis, transpose_if_needed
from thesis_lib.transforms import rmse_z, zscore_using_true_stats
from utils.classification import load_precomputed_results

logger = logging.getLogger(__name__)

# Avoid repeating the same out-of-range warning for every trial index during thesis HTML gen.
_oof_trial_warned: set[str] = set()


from thesis_lib.errors import ThesisDataError

InputMode = Literal["neural", "behavioral"]


@dataclass(frozen=True)
class TrialZSeries:
    """Z_true and model Zp predictions for one trial and one output channel.

    Any model trace may be all-NaN when that model does not have the requested
    trial index (e.g. VARMA with fewer test rows than PSID/DPAD).
    """

    t_abs: np.ndarray
    z_true_raw: np.ndarray
    z_psid: np.ndarray
    z_dpad: np.ndarray
    z_varma: np.ndarray


def load_split_results(
    results_root: Path,
    variant: str,
    run_timestamp: str,
    split: str,
) -> Optional[Dict[str, Any]]:
    variant_dir = results_root / variant
    return load_precomputed_results(variant_dir, run_timestamp, split)


def load_split_results_required(
    results_root: Path,
    variant: str,
    run_timestamp: str,
    split: str,
) -> Dict[str, Any]:
    """
    Load split results or raise ``ThesisDataError`` with the expected parquet / pickle path.

    Thesis HTML uses this instead of silent timestamp fallbacks so missing runs surface immediately.
    """
    r = load_split_results(results_root, variant, run_timestamp, split)
    if r is not None:
        return r
    variant_dir = results_root / variant
    pq = variant_dir / split / f"test_results_{run_timestamp}.parquet"
    pkl = variant_dir / run_timestamp / f"{split}_results.pkl"
    msg = (
        f"No results for variant={variant!r} run_ts={run_timestamp!r} split={split!r}. "
        f"Expected parquet directory or file: {pq} (or pickle {pkl})."
    )
    raise ThesisDataError(msg)


def _latest_test_parquet_ts(split_dir: Path) -> Optional[str]:
    """Backward-compatible name; see ``utils.thesis_result_timestamps.latest_test_parquet_ts``."""
    from utils.thesis_result_timestamps import latest_test_parquet_ts

    return latest_test_parquet_ts(split_dir)


def latest_run_timestamp_on_disk(
    variant_dir: Path, *, split: str = "test"
) -> tuple[Optional[str], str]:
    """See ``utils.thesis_result_timestamps.latest_run_timestamp_on_disk``."""
    from utils.thesis_result_timestamps import latest_run_timestamp_on_disk as _latest

    return _latest(variant_dir, split=split)


def load_split_results_with_fallback(
    results_root: Path,
    variant: str,
    preferred_run_ts: str,
    split: str,
) -> Optional[Dict[str, Any]]:
    """
    Load split results using ``preferred_run_ts``; if that run has no data for this variant,
    retry with the latest ``model_*.pkl`` timestamp, then the latest test parquet timestamp.

    Cross-eval directories (e.g. ``dbs_on_eval_off``) have no ``model_*.pkl`` — in that case
    we fall back to the newest ``test_results_<ts>.parquet`` inside the split subdirectory.
    """
    r = load_split_results(results_root, variant, preferred_run_ts, split)
    if r is not None:
        return r
    variant_dir = results_root / variant
    if not variant_dir.is_dir():
        return None

    alt: Optional[str] = None
    try:
        from utils.miscellaneous import get_latest_timestamp

        alt = get_latest_timestamp(str(variant_dir))
    except (ValueError, StopIteration):
        pass

    if alt is None:
        alt = _latest_test_parquet_ts(variant_dir / split)

    if alt is None or alt == preferred_run_ts:
        return None
    return load_split_results(results_root, variant, alt, split)


def output_channel_label(
    split_res: Optional[Dict[str, Any]],
    channel_idx: int,
    *,
    fallback: str = "",
    preserve_underscores: bool = False,
) -> str:
    """
    Human-readable output channel name from saved ``output_channels`` (trainer metadata).

    Returns ``fallback`` when missing or out of range. By default underscores become spaces;
    set ``preserve_underscores=True`` for names like ``ECoG_1``.
    """
    if split_res is None or channel_idx < 0:
        return fallback
    raw = split_res.get("output_channels")
    if raw is None:
        return fallback
    seq: Sequence[Any]
    if isinstance(raw, np.ndarray):
        seq = raw.tolist()
    elif isinstance(raw, (list, tuple)):
        seq = raw
    else:
        try:
            seq = list(raw)  # type: ignore[arg-type]
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


def channels_as_str_list(raw: Any) -> list[str]:
    """Normalize ``output_channels`` / ``input_channels`` metadata to a list of strings."""
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        seq = raw.tolist()
    elif isinstance(raw, (list, tuple)):
        seq = raw
    else:
        try:
            seq = list(raw)  # type: ignore[arg-type]
        except TypeError:
            return []
    out: list[str] = []
    for item in seq:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def format_channels_for_caption(names: Sequence[str], *, max_items: int = 45) -> str:
    """Comma-separated channel names; truncates with count if very long."""
    names = [str(n).replace("_", " ") for n in names if str(n).strip()]
    if not names:
        return ""
    if len(names) <= max_items:
        return ", ".join(names)
    head = ", ".join(names[:max_items])
    return f"{head}, … (+{len(names) - max_items} more)"


def resolve_output_channel_display(
    split_res: Dict[str, Any],
    channel_idx: int,
    *,
    declared_outputs: Sequence[str],
) -> tuple[str, bool]:
    """
    Human-readable name for output channel ``channel_idx``.

    Returns (label, used_declared_only). If parquet lacks ``output_channels`` but
    ``declared_outputs`` has ``channel_idx``, the declared name is used and the second
    value is True (caller should disclose in captions).
    """
    meta = channels_as_str_list(split_res.get("output_channels"))
    if channel_idx < len(meta):
        return meta[channel_idx], False
    if channel_idx < 0:
        raise ThesisDataError(f"Invalid channel_idx={channel_idx}")
    if channel_idx < len(declared_outputs):
        return str(declared_outputs[channel_idx]), True
    raise ThesisDataError(
        f"output_channels missing in results and channel_idx={channel_idx} "
        f"out of range for declared_outputs ({len(declared_outputs)} entries). "
        "Re-save test results with output_channels or extend THESIS_DECLARED_BEHAVIORAL_OUTPUTS in specs."
    )


def require_input_channels_list(
    split_res: Dict[str, Any], *, context: str
) -> list[str]:
    """Neural / covariate input names from saved metadata; raises if absent or empty."""
    names = channels_as_str_list(split_res.get("input_channels"))
    if not names:
        raise ThesisDataError(
            f"{context}: missing or empty input_channels in saved results. "
            "Re-run the tester so metadata is written to the split parquet."
        )
    return names


def describe_saved_io_plaintext(
    split_res: Dict[str, Any],
    *,
    declared_outputs: Sequence[str],
    max_neural_list: int = 40,
) -> str:
    """
    Multi-line plaintext: behavioral outputs + neural inputs (explicit lists or errors inline).

    Uses metadata when present; output names fall back to ``declared_outputs`` with a clear tag.
    """
    lines: list[str] = []
    meta_out = channels_as_str_list(split_res.get("output_channels"))
    if meta_out:
        lines.append(
            "Behavioral outputs (from saved metadata): "
            + format_channels_for_caption(meta_out)
        )
    elif declared_outputs:
        lines.append(
            "Behavioral outputs (thesis default — parquet lacked output_channels): "
            + format_channels_for_caption(list(declared_outputs))
        )
    else:
        lines.append("Behavioral outputs: not specified.")

    inc = split_res.get("input_channels")
    inn = channels_as_str_list(inc)
    if inn:
        lines.append(
            "Neural / model inputs (from saved metadata, narrow-band features): "
            + format_channels_for_caption(inn, max_items=max_neural_list)
        )
    else:
        lines.append(
            "Neural / model inputs: not listed in saved metadata (input_channels missing or empty)."
        )
    return "\n".join(lines)


def short_io_caption(
    split_res: Dict[str, Any],
    *,
    declared_outputs: Sequence[str],
    max_outputs: int = 6,
    max_inputs: int = 8,
) -> str:
    """Single-line behavioral outputs + neural inputs for thesis figure captions."""
    meta_out = channels_as_str_list(split_res.get("output_channels"))
    inn = channels_as_str_list(split_res.get("input_channels"))
    parts = []
    if meta_out:
        parts.append(
            f"Outputs: {format_channels_for_caption(meta_out, max_items=max_outputs)}"
        )
    elif declared_outputs:
        parts.append(
            "Outputs (thesis default): "
            + format_channels_for_caption(list(declared_outputs), max_items=max_outputs)
        )
    else:
        parts.append("Outputs: not specified.")
    if inn:
        parts.append(
            f"Inputs: {format_channels_for_caption(inn, max_items=max_inputs)}"
        )
    else:
        parts.append("Inputs: not listed in saved metadata.")
    return " ".join(parts)


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


def trial_descriptor_text(split_res: Dict[str, Any], trial_idx: int) -> str:
    """Participant / session / block / trial for one PSID-aligned test row."""

    def _one(key: str) -> str:
        seq = split_res.get(key)
        if seq is None or trial_idx >= len(seq):
            return "?"
        x = seq[trial_idx]
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
            x = x[0]
        return str(x)

    return (
        f"Participant {_one('participant_id')}, session {_one('session')}, "
        f"block {_one('block')}, trial {_one('trial')} "
        f"(PSID test row index {trial_idx})"
    )


def _trial_field_str(split_res: Dict[str, Any], trial_idx: int, key: str) -> str:
    seq = split_res.get(key)
    if seq is None or trial_idx >= len(seq):
        return "?"
    x = seq[trial_idx]
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
        x = x[0]
    return str(x)


def resolve_neural_y_channel_idx(
    split_res: Dict[str, Any],
    neural_y_feature_name: str,
    fallback_channel_idx: int,
) -> int:
    """Match YAML ``neural_input`` string to ``input_channels``."""
    raw = (neural_y_feature_name or "").strip()
    if not raw:
        return int(fallback_channel_idx)
    names = channels_as_str_list(split_res.get("input_channels"))
    if not names:
        # 200Hz narrow_band test parquets store an empty input_channels list.
        # Fall back to the caller's declared channel_idx (matches sister fn in
        # notebooks/thesis_loaders.py).
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


def split_res_with_nonempty_input_channels(
    *candidates: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """First candidate whose ``input_channels`` is non-empty (saved training feature order)."""
    for split_res in candidates:
        if split_res is None:
            continue
        if channels_as_str_list(split_res.get("input_channels")):
            return split_res
    return None


def resolve_neural_y_channel_idx_from_candidates(
    neural_y_feature_name: str,
    fallback_channel_idx: int,
    *candidates: Optional[Dict[str, Any]],
) -> int:
    """
    Like ``resolve_neural_y_channel_idx``, but try several result dicts (e.g. PSID, DPAD, VARMA joint).

    Some checkpoints omit ``input_channels`` in the pickle; another framework's aligned joint often has it.
    """
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


def neural_y_feature_label(
    split_res: Dict[str, Any],
    channel_idx: int,
    *,
    neural_y_feature_name: str = "",
) -> str:
    """
    Human-readable neural target name for captions / axes.

    Prefer the column in saved ``input_channels`` (training order). If ``neural_y_feature_name`` is set
    (YAML ``neural_input``), resolve index with ``resolve_neural_y_channel_idx``; if the name is missing
    from metadata, show that string. Fall back to ``output_channels``, then ``output_channel_label``.
    """
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
    if lab:
        return lab
    return f"neural Y column {ch}"


def thesis_exemplar_tagline(
    split_res: Dict[str, Any],
    i_off: int,
    i_on: int,
    feature_label: str,
    *,
    participant_label: str = "",
) -> str:
    """One-line PDI/session/block/trial (with row indices) + feature label for thesis captions."""
    pid = (participant_label or "").strip() or _trial_field_str(
        split_res, i_off, "participant_id"
    )
    sess = _trial_field_str(split_res, i_off, "session")

    def arm(i: int, lbl: str) -> str:
        b = _trial_field_str(split_res, i, "block")
        t = _trial_field_str(split_res, i, "trial")
        return f"{lbl}: block {b}, trial {t} (row {i})"

    return f"{pid} · session {sess} · {arm(i_off, 'OFF')} · {arm(i_on, 'ON')} · {feature_label}"


def extract_trial_z_series(
    split_res_psid: Dict[str, Any],
    split_res_dpad: Optional[Dict[str, Any]],
    split_res_varma: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
    varma_trial_idx: Optional[int] = None,
) -> TrialZSeries:
    """
    Extract raw Z and Zp from three result dicts.

    PSID is required (it provides the ground-truth time axis). DPAD and VARMA
    are optional: when a model lacks the requested trial index its trace is
    filled with NaN so downstream code can still render the remaining models.

    varma_trial_idx: when set, use this index for VARMA instead of trial_idx
    (e.g. when VARMA comes from dbs_off/dbs_on with different trial order).
    """
    v_idx = varma_trial_idx if varma_trial_idx is not None else trial_idx
    if not _model_has_trial(split_res_psid, trial_idx):
        n_psid = _n_z_trials(split_res_psid)
        raise ValueError(
            f"PSID: missing Z/Zp for trial_idx={trial_idx} (n_trials={n_psid}). "
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
                    "%s: trial index out of range (n_trials=%d vs PSID rows) — "
                    "traces set to NaN; align VARMA/DPAD test parquets with PSID.",
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

    def _varma() -> np.ndarray:
        return _zp_chan_or_nan("VARMA", split_res_varma, v_idx)

    return TrialZSeries(
        t_abs=t_abs,
        z_true_raw=true_c,
        z_psid=_zp_chan_or_nan("PSID", split_res_psid, trial_idx),
        z_dpad=_zp_chan_or_nan("DPAD", split_res_dpad, trial_idx),
        z_varma=_varma(),
    )


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


def extract_trial_y_series(
    split_res_psid: Dict[str, Any],
    split_res_dpad: Optional[Dict[str, Any]],
    split_res_varma: Dict[str, Any],
    trial_idx: int,
    y_channel_idx: int,
    varma_trial_idx: Optional[int] = None,
) -> TrialZSeries:
    """
    Like ``extract_trial_z_series`` but uses neural targets ``Y`` / ``Yp`` (same trial alignment).
    """
    v_idx = varma_trial_idx if varma_trial_idx is not None else trial_idx
    if not _model_has_trial_y(split_res_psid, trial_idx):
        n_psid = len(split_res_psid.get("Y") or [])
        raise ValueError(
            f"PSID: missing Y/Yp for trial_idx={trial_idx} (n_trials={n_psid}). "
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
                    "%s: Y/Yp trial index out of range (n_trials=%d) — traces set to NaN; align test parquets.",
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

    def _varma_y() -> np.ndarray:
        return _yp_chan_or_nan("VARMA", split_res_varma, v_idx)

    return TrialZSeries(
        t_abs=t_abs,
        z_true_raw=true_c,
        z_psid=_yp_chan_or_nan("PSID", split_res_psid, trial_idx),
        z_dpad=_yp_chan_or_nan("DPAD", split_res_dpad, trial_idx),
        z_varma=_varma_y(),
    )


def trial_rmse_z_for_model(
    split_res: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
) -> float:
    """Single-model RMSE on z-scored output: sqrt(mean((z_true - z_pred)^2)) for one trial."""
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
    """Single-model RMSE on z-scored neural Y: same z-scoring rule as behavioral Z."""
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
