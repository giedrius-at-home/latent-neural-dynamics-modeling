"""Helpers for thesis exemplar trial selection (e.g. adjacent OFF → ON across block boundaries)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dashboard.thesis.aggregate_rmse import normalize_stim


def _as_list(seq: Any) -> Optional[List[Any]]:
    if seq is None:
        return None
    if isinstance(seq, np.ndarray):
        return seq.tolist()
    return list(seq)


def _session_cell(seq: Optional[Sequence[Any]], i: int) -> str:
    if seq is None or i >= len(seq):
        return ""
    x = seq[i]
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
        x = x[0]
    return str(x).strip()


def find_adjacent_off_then_on_trial_indices(
    stim_list: Optional[Sequence[Any]],
) -> Optional[Tuple[int, int]]:
    """
    First index pair (i, i+1) with DBS-OFF then DBS-ON in saved trial order.

    Returns None if stim is missing or no such adjacent transition exists.
    """
    if stim_list is None:
        return None
    n = len(stim_list)
    if n < 2:
        return None
    for i in range(n - 1):
        a = normalize_stim(stim_list[i])
        b = normalize_stim(stim_list[i + 1])
        if a == "off" and b == "on":
            return int(i), int(i + 1)
    return None


def find_block_boundary_off_then_on_trial_indices(
    split_res: Dict[str, Any],
) -> Optional[Tuple[int, int]]:
    """
    First (i, i+1) where trial i is DBS-OFF, i+1 is DBS-ON, **and** they sit on a **block**
    boundary (different ``block`` id) within the **same session**.

    This targets “last trial of the OFF block / first trial of the ON block” when the run
    table is sorted by participant, session, block, trial (trainer order).

    If ``block`` metadata is missing, falls back to the first adjacent OFF→ON in ``stim``
    order (same as :func:`find_adjacent_off_then_on_trial_indices`).
    """
    stim = _as_list(split_res.get("stim"))
    if not stim or len(stim) < 2:
        return None
    block = _as_list(split_res.get("block"))
    session = split_res.get("session")

    for i in range(len(stim) - 1):
        a = normalize_stim(stim[i])
        b = normalize_stim(stim[i + 1])
        if a != "off" or b != "on":
            continue
        if block is not None and len(block) > i + 1:
            if block[i] == block[i + 1]:
                continue
        if session is not None:
            s0 = _session_cell(session, i)
            s1 = _session_cell(session, i + 1)
            if s0 and s1 and s0 != s1:
                continue
        return int(i), int(i + 1)

    return None


def resolve_off_on_indices_from_spec(
    *,
    trial_idx_off: int,
    trial_idx_on: int,
    use_adjacent_off_on_trials: bool,
    split_res: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Return (off, on) trial row indices, optionally overriding with block-edge OFF→ON.

    Preference order: (1) OFF→ON with different ``block`` id and same session;
    (2) first adjacent OFF→ON in ``stim`` order (when (1) is unavailable);
    if neither exists, raises so the caller can skip the figure.
    """
    if not use_adjacent_off_on_trials:
        return trial_idx_off, trial_idx_on
    pair = find_block_boundary_off_then_on_trial_indices(split_res)
    if pair is not None:
        return pair
    stim = _as_list(split_res.get("stim"))
    pair = find_adjacent_off_then_on_trial_indices(stim)
    if pair is None:
        raise ValueError(
            "use_adjacent_off_on_trials=True but no OFF→ON transition found "
            "(block boundary or adjacent stim rows)."
        )
    return pair
