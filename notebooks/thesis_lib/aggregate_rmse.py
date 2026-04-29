"""
Pool per-trial RMSE (z-scored output) across aligned PSID/DPAD/VARMA triplets, then aggregate
mean ± SEM per model × DBS cell. Optional paired Wilcoxon vs VARMA baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from thesis_lib.loaders import (
    ThesisDataError,
    load_split_results,
    load_split_results_required,
    trial_rmse_z_for_model,
)

if TYPE_CHECKING:
    from thesis_lib.specs import AlignedTriplet

logger = logging.getLogger(__name__)

TrialKey = Tuple[Any, Any, Any, Any]


def normalize_stim(val: Any) -> Optional[str]:
    """Return 'on' or 'off', or None if unknown."""
    if val is None:
        return None
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
        val = val[0]
    s = str(val).strip().lower()
    if s in ("on", "1", "true"):
        return "on"
    if s in ("off", "0", "false"):
        return "off"
    return None


def _trial_key(split_res: Dict[str, Any], trial_idx: int) -> TrialKey:
    pid = split_res["participant_id"][trial_idx]
    sess = split_res["session"][trial_idx]
    blk = split_res["block"][trial_idx]
    tri = split_res["trial"][trial_idx]

    def _one(x: Any) -> str:
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
            x = x[0]
        return str(x)

    return (_one(pid), _one(sess), _one(blk), _one(tri))


def _key_index_map(split_res: Dict[str, Any]) -> Dict[TrialKey, int]:
    zl = split_res.get("Z")
    if not zl:
        return {}
    out: Dict[TrialKey, int] = {}
    for i in range(len(zl)):
        if zl[i] is None:
            continue
        k = _trial_key(split_res, i)
        out[k] = i
    return out


def _sem(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 2:
        return float("nan")
    return float(np.std(a, ddof=1) / np.sqrt(n))


def _wilcoxon_paired(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Paired Wilcoxon signed-rank; returns p-value or None if test cannot run."""
    from scipy import stats

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = min(x.size, y.size)
    if m < 6:
        return None
    x, y = x[:m], y[:m]
    d = x - y
    if np.allclose(d, 0):
        return None
    try:
        try:
            res = stats.wilcoxon(x, y, alternative="two-sided", method="auto")
        except TypeError:
            res = stats.wilcoxon(x, y, alternative="two-sided")
        p = float(res.pvalue)
        return p if np.isfinite(p) else None
    except Exception as e:
        logger.debug("Wilcoxon skipped: %s", e)
        return None


def p_to_stars(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


@dataclass
class WilcoxonResults:
    psid_vs_varma_off_p: Optional[float] = None
    psid_vs_varma_on_p: Optional[float] = None
    dpad_vs_varma_off_p: Optional[float] = None
    dpad_vs_varma_on_p: Optional[float] = None
    psid_vs_dpad_off_p: Optional[float] = None
    psid_vs_dpad_on_p: Optional[float] = None


@dataclass
class WithinCrossRmseData:
    """
    Per-model, per-condition (OFF/ON), within and cross trial RMSE lists.
    Structure: models = ["PSID", "DPAD", "VARMA"]; each has (within_vals, cross_vals) for OFF and ON.
    """

    psid_off_within: List[float] = field(default_factory=list)
    psid_off_cross: List[float] = field(default_factory=list)
    psid_on_within: List[float] = field(default_factory=list)
    psid_on_cross: List[float] = field(default_factory=list)
    dpad_off_within: List[float] = field(default_factory=list)
    dpad_off_cross: List[float] = field(default_factory=list)
    dpad_on_within: List[float] = field(default_factory=list)
    dpad_on_cross: List[float] = field(default_factory=list)
    varma_off_within: List[float] = field(default_factory=list)
    varma_off_cross: List[float] = field(default_factory=list)
    varma_on_within: List[float] = field(default_factory=list)
    varma_on_cross: List[float] = field(default_factory=list)
    n_triplets_used: int = 0


def collect_within_cross_rmse(
    results_root: Path,
    triplet_specs: List["AlignedTriplet"],
    channel_idx: int,
    split: str = "test",
) -> WithinCrossRmseData:
    """
    Collect trial RMSE for within (same-condition model) and cross (opposite-condition model).
    Within OFF: dbs_off model on OFF trials. Cross OFF: dbs_on model eval on OFF trials.
    """
    from thesis_lib.specs import (
        _variant_cross_eval,
        _variant_off,
        _variant_on,
        triplet_branch_timestamp,
    )

    out = WithinCrossRmseData()
    n_ok = 0

    for tri in triplet_specs:
        psid_off = load_split_results_required(
            results_root,
            _variant_off(tri.psid_variant),
            triplet_branch_timestamp(tri, "psid", "off"),
            split,
        )
        psid_on = load_split_results_required(
            results_root,
            _variant_on(tri.psid_variant),
            triplet_branch_timestamp(tri, "psid", "on"),
            split,
        )
        psid_on_eval_off = load_split_results_required(
            results_root,
            _variant_cross_eval(_variant_on(tri.psid_variant), "off"),
            triplet_branch_timestamp(tri, "psid", "eval_off"),
            split,
        )
        psid_off_eval_on = load_split_results_required(
            results_root,
            _variant_cross_eval(_variant_off(tri.psid_variant), "on"),
            triplet_branch_timestamp(tri, "psid", "eval_on"),
            split,
        )
        _dpad_available = bool(tri.dpad_run_ts_off and tri.dpad_run_ts_on)
        if _dpad_available:
            dpad_off = load_split_results(
                results_root,
                _variant_off(tri.dpad_variant),
                triplet_branch_timestamp(tri, "dpad", "off"),
                split,
            )
            dpad_on = load_split_results(
                results_root,
                _variant_on(tri.dpad_variant),
                triplet_branch_timestamp(tri, "dpad", "on"),
                split,
            )
            dpad_on_eval_off = load_split_results(
                results_root,
                _variant_cross_eval(_variant_on(tri.dpad_variant), "off"),
                triplet_branch_timestamp(tri, "dpad", "eval_off"),
                split,
            )
            dpad_off_eval_on = load_split_results(
                results_root,
                _variant_cross_eval(_variant_off(tri.dpad_variant), "on"),
                triplet_branch_timestamp(tri, "dpad", "eval_on"),
                split,
            )
        else:
            dpad_off = dpad_on = dpad_on_eval_off = dpad_off_eval_on = None
        varma_off = load_split_results_required(
            results_root,
            _variant_off(tri.varma_variant),
            triplet_branch_timestamp(tri, "varma", "off"),
            split,
        )
        varma_on = load_split_results_required(
            results_root,
            _variant_on(tri.varma_variant),
            triplet_branch_timestamp(tri, "varma", "on"),
            split,
        )
        varma_on_eval_off = load_split_results_required(
            results_root,
            _variant_cross_eval(_variant_on(tri.varma_variant), "off"),
            triplet_branch_timestamp(tri, "varma", "eval_off"),
            split,
        )
        varma_off_eval_on = load_split_results_required(
            results_root,
            _variant_cross_eval(_variant_off(tri.varma_variant), "on"),
            triplet_branch_timestamp(tri, "varma", "eval_on"),
            split,
        )

        mp_off = _key_index_map(psid_off)
        mp_on = _key_index_map(psid_on)
        common_off = set(mp_off.keys())
        common_on = set(mp_on.keys())

        for k in sorted(
            common_off, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))
        ):
            i = mp_off[k]
            try:
                r = trial_rmse_z_for_model(psid_off, i, channel_idx)
                out.psid_off_within.append(r)
            except Exception:
                continue
            i_cross = (
                _key_index_map(psid_on_eval_off).get(k) if psid_on_eval_off else None
            )
            if i_cross is not None:
                try:
                    out.psid_off_cross.append(
                        trial_rmse_z_for_model(psid_on_eval_off, i_cross, channel_idx)
                    )
                except Exception:
                    pass

            i_d = _key_index_map(dpad_off).get(k) if dpad_off else None
            if i_d is not None:
                try:
                    out.dpad_off_within.append(
                        trial_rmse_z_for_model(dpad_off, i_d, channel_idx)
                    )
                except Exception:
                    pass
            i_d_cross = (
                _key_index_map(dpad_on_eval_off).get(k) if dpad_on_eval_off else None
            )
            if i_d_cross is not None:
                try:
                    out.dpad_off_cross.append(
                        trial_rmse_z_for_model(dpad_on_eval_off, i_d_cross, channel_idx)
                    )
                except Exception:
                    pass

            i_v = _key_index_map(varma_off).get(k) if varma_off else None
            if i_v is not None:
                try:
                    out.varma_off_within.append(
                        trial_rmse_z_for_model(varma_off, i_v, channel_idx)
                    )
                except Exception:
                    pass
            i_v_cross = (
                _key_index_map(varma_on_eval_off).get(k) if varma_on_eval_off else None
            )
            if i_v_cross is not None:
                try:
                    out.varma_off_cross.append(
                        trial_rmse_z_for_model(
                            varma_on_eval_off, i_v_cross, channel_idx
                        )
                    )
                except Exception:
                    pass

        for k in sorted(
            common_on, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))
        ):
            i = mp_on[k]
            try:
                r = trial_rmse_z_for_model(psid_on, i, channel_idx)
                out.psid_on_within.append(r)
            except Exception:
                continue
            i_cross = (
                _key_index_map(psid_off_eval_on).get(k) if psid_off_eval_on else None
            )
            if i_cross is not None:
                try:
                    out.psid_on_cross.append(
                        trial_rmse_z_for_model(psid_off_eval_on, i_cross, channel_idx)
                    )
                except Exception:
                    pass

            i_d = _key_index_map(dpad_on).get(k) if dpad_on else None
            if i_d is not None:
                try:
                    out.dpad_on_within.append(
                        trial_rmse_z_for_model(dpad_on, i_d, channel_idx)
                    )
                except Exception:
                    pass
            i_d_cross = (
                _key_index_map(dpad_off_eval_on).get(k) if dpad_off_eval_on else None
            )
            if i_d_cross is not None:
                try:
                    out.dpad_on_cross.append(
                        trial_rmse_z_for_model(dpad_off_eval_on, i_d_cross, channel_idx)
                    )
                except Exception:
                    pass

            i_v = _key_index_map(varma_on).get(k) if varma_on else None
            if i_v is not None:
                try:
                    out.varma_on_within.append(
                        trial_rmse_z_for_model(varma_on, i_v, channel_idx)
                    )
                except Exception:
                    pass
            i_v_cross = (
                _key_index_map(varma_off_eval_on).get(k) if varma_off_eval_on else None
            )
            if i_v_cross is not None:
                try:
                    out.varma_on_cross.append(
                        trial_rmse_z_for_model(
                            varma_off_eval_on, i_v_cross, channel_idx
                        )
                    )
                except Exception:
                    pass

        n_ok += 1

    if n_ok == 0:
        raise ThesisDataError(
            "collect_within_cross_rmse: no triplets produced usable PSID off/on alignment."
        )
    out.n_triplets_used = n_ok
    return out


@dataclass
class AggregateRmseData:
    """Six cells in order: PSID off, PSID on, DPAD off, DPAD on, VARMA off, VARMA on."""

    labels: Tuple[str, ...] = (
        "PSID DBS-OFF",
        "PSID DBS-ON",
        "DPAD DBS-OFF",
        "DPAD DBS-ON",
        "VARMA DBS-OFF",
        "VARMA DBS-ON",
    )
    means: Tuple[float, ...] = (0.0,) * 6
    sems: Tuple[float, ...] = (0.0,) * 6
    trial_rmse: Tuple[List[float], ...] = tuple([] for _ in range(6))
    # (rmse, participant_id) per cell for box-plot participant coloring
    trial_rmse_with_participant: Tuple[List[Tuple[float, str]], ...] = tuple(
        [] for _ in range(6)
    )
    wilcoxon: WilcoxonResults = field(default_factory=WilcoxonResults)
    n_triplets_used: int = 0


def collect_pooled_rmse(
    results_root: Path,
    triplet_specs: List["AlignedTriplet"],
    channel_idx: int,
    split: str = "test",
    run_wilcoxon: bool = True,
) -> AggregateRmseData:
    """
    Each `AlignedTriplet` lists matching PSID / DPAD / VARMA variant names and run timestamps
    for one participant/session slice.
    """
    cells: List[List[float]] = [[] for _ in range(6)]
    cells_with_pid: List[List[Tuple[float, str]]] = [[] for _ in range(6)]

    paired_off_psid_varma: List[Tuple[float, float]] = []
    paired_on_psid_varma: List[Tuple[float, float]] = []
    paired_off_dpad_varma: List[Tuple[float, float]] = []
    paired_on_dpad_varma: List[Tuple[float, float]] = []
    paired_off_psid_dpad: List[Tuple[float, float]] = []
    paired_on_psid_dpad: List[Tuple[float, float]] = []

    if not triplet_specs:
        raise ThesisDataError("collect_pooled_rmse: triplet_specs is empty.")

    n_ok = 0
    for tri in triplet_specs:
        res_p = load_split_results_required(
            results_root, tri.psid_variant, tri.psid_run_ts, split
        )
        res_v = load_split_results_required(
            results_root, tri.varma_variant, tri.varma_run_ts, split
        )
        has_dpad = bool(tri.dpad_run_ts)
        res_d = (
            load_split_results(results_root, tri.dpad_variant, tri.dpad_run_ts, split)
            if has_dpad
            else None
        )

        mp = _key_index_map(res_p)
        mv = _key_index_map(res_v)
        md = _key_index_map(res_d) if res_d else {}
        common_pd = set(mp.keys()) & set(mv.keys())
        if md:
            common_pd &= set(md.keys())
        if not common_pd:
            raise ThesisDataError(
                f"Pooled RMSE triplet {tri.label!r}: no trial keys common to PSID and VARMA."
            )

        n_ok += 1
        for k in sorted(
            common_pd, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))
        ):
            i_p, i_v = mp[k], mv[k]
            stim = normalize_stim(res_p["stim"][i_p])
            if stim is None:
                continue
            try:
                r_p = trial_rmse_z_for_model(res_p, i_p, channel_idx)
                r_d = (
                    trial_rmse_z_for_model(res_d, md[k], channel_idx)
                    if res_d and k in md
                    else float("nan")
                )
                r_v = trial_rmse_z_for_model(res_v, i_v, channel_idx)
            except Exception as e:
                logger.debug("Skip trial %s: %s", k, e)
                continue

            pid = str(k[0]) if k else "?"
            if stim == "off":
                cells[0].append(r_p)
                cells[2].append(r_d)
                cells_with_pid[0].append((r_p, pid))
                cells_with_pid[2].append((r_d, pid))
                cells[4].append(r_v)
                cells_with_pid[4].append((r_v, pid))
                paired_off_psid_varma.append((r_p, r_v))
                paired_off_dpad_varma.append((r_d, r_v))
                paired_off_psid_dpad.append((r_p, r_d))
            else:
                cells[1].append(r_p)
                cells[3].append(r_d)
                cells_with_pid[1].append((r_p, pid))
                cells_with_pid[3].append((r_d, pid))
                cells[5].append(r_v)
                cells_with_pid[5].append((r_v, pid))
                paired_on_psid_varma.append((r_p, r_v))
                paired_on_dpad_varma.append((r_d, r_v))
                paired_on_psid_dpad.append((r_p, r_d))

    if n_ok == 0:
        raise ThesisDataError(
            "collect_pooled_rmse: no triplets produced aligned PSID/DPAD/VARMA trial keys."
        )
    if sum(len(c) for c in cells) == 0:
        raise ThesisDataError(
            "collect_pooled_rmse: no trial RMSE values after alignment (check stim and Z/Zp)."
        )

    means = tuple(float(np.mean(c)) if len(c) else float("nan") for c in cells)
    sems = tuple(_sem(np.array(c)) if len(c) > 1 else float("nan") for c in cells)

    w = WilcoxonResults()
    if run_wilcoxon:
        if paired_off_psid_varma:
            x, y = zip(*paired_off_psid_varma)
            w.psid_vs_varma_off_p = _wilcoxon_paired(np.array(x), np.array(y))
        if paired_on_psid_varma:
            x, y = zip(*paired_on_psid_varma)
            w.psid_vs_varma_on_p = _wilcoxon_paired(np.array(x), np.array(y))
        if paired_off_dpad_varma:
            x, y = zip(*paired_off_dpad_varma)
            w.dpad_vs_varma_off_p = _wilcoxon_paired(np.array(x), np.array(y))
        if paired_on_dpad_varma:
            x, y = zip(*paired_on_dpad_varma)
            w.dpad_vs_varma_on_p = _wilcoxon_paired(np.array(x), np.array(y))
        if paired_off_psid_dpad:
            x, y = zip(*paired_off_psid_dpad)
            w.psid_vs_dpad_off_p = _wilcoxon_paired(np.array(x), np.array(y))
        if paired_on_psid_dpad:
            x, y = zip(*paired_on_psid_dpad)
            w.psid_vs_dpad_on_p = _wilcoxon_paired(np.array(x), np.array(y))

    return AggregateRmseData(
        means=means,
        sems=sems,
        trial_rmse=tuple(cells),
        trial_rmse_with_participant=tuple(cells_with_pid),
        wilcoxon=w,
        n_triplets_used=n_ok,
    )
