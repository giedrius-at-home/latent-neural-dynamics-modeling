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

from dashboard.thesis.loaders import load_split_results, trial_rmse_z_for_model

if TYPE_CHECKING:
    from dashboard.thesis.specs import AlignedTriplet

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

    def _one(x: Any) -> Any:
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
            return x[0]
        return x

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


@dataclass
class AggregateRmseData:
    """Six cells in order: PSID off, PSID on, DPAD off, DPAD on, VARMA off, VARMA on."""

    labels: Tuple[str, ...] = (
        "PSID DBS-OFF",
        "PSID DBS-ON",
        "DPAD-RNN DBS-OFF",
        "DPAD-RNN DBS-ON",
        "VARMA DBS-OFF",
        "VARMA DBS-ON",
    )
    means: Tuple[float, ...] = (0.0,) * 6
    sems: Tuple[float, ...] = (0.0,) * 6
    trial_rmse: Tuple[List[float], ...] = tuple([] for _ in range(6))
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

    paired_off_psid_varma: List[Tuple[float, float]] = []
    paired_on_psid_varma: List[Tuple[float, float]] = []
    paired_off_dpad_varma: List[Tuple[float, float]] = []
    paired_on_dpad_varma: List[Tuple[float, float]] = []

    n_ok = 0
    for tri in triplet_specs:
        res_p = load_split_results(results_root, tri.psid_variant, tri.psid_run_ts, split)
        res_d = load_split_results(results_root, tri.dpad_variant, tri.dpad_run_ts, split)
        res_v = load_split_results(results_root, tri.varma_variant, tri.varma_run_ts, split)
        if res_p is None or res_d is None or res_v is None:
            logger.warning(
                "Skipping triplet %s: missing results (psid=%s, dpad=%s, varma=%s)",
                getattr(tri, "label", ""),
                res_p is not None,
                res_d is not None,
                res_v is not None,
            )
            continue

        mp = _key_index_map(res_p)
        md = _key_index_map(res_d)
        mv = _key_index_map(res_v)
        common = set(mp.keys()) & set(md.keys()) & set(mv.keys())
        if not common:
            logger.warning(
                "Skipping triplet %s: no overlapping trial keys across PSID/DPAD/VARMA.",
                getattr(tri, "label", ""),
            )
            continue

        n_ok += 1
        for k in sorted(common, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))):
            i_p, i_d, i_v = mp[k], md[k], mv[k]
            stim = normalize_stim(res_p["stim"][i_p])
            if stim is None:
                continue
            try:
                r_p = trial_rmse_z_for_model(res_p, i_p, channel_idx)
                r_d = trial_rmse_z_for_model(res_d, i_d, channel_idx)
                r_v = trial_rmse_z_for_model(res_v, i_v, channel_idx)
            except Exception as e:
                logger.debug("Skip trial %s: %s", k, e)
                continue

            if stim == "off":
                cells[0].append(r_p)
                cells[2].append(r_d)
                cells[4].append(r_v)
                paired_off_psid_varma.append((r_p, r_v))
                paired_off_dpad_varma.append((r_d, r_v))
            else:
                cells[1].append(r_p)
                cells[3].append(r_d)
                cells[5].append(r_v)
                paired_on_psid_varma.append((r_p, r_v))
                paired_on_dpad_varma.append((r_d, r_v))

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

    return AggregateRmseData(
        means=means,
        sems=sems,
        trial_rmse=tuple(cells),
        wilcoxon=w,
        n_triplets_used=n_ok,
    )
