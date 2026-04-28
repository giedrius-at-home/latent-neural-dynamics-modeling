"""Shared config for thesis appendix figures: style, colors, data loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from thesis_lib.constants import COLOR_DBS_OFF, COLOR_DBS_ON
from thesis_lib.loaders import load_split_results
from thesis_lib.specs import _ALL_TRIPLETS

# DBS colours (match thesis constants)
COLORS = {
    "dbs_off": COLOR_DBS_OFF,
    "dbs_on": COLOR_DBS_ON,
}

# Figure styling
STYLE = {
    "figsize_grid22": (8, 7),
    "figsize_grid14": (12, 3.5),
    "figsize_grid41": (6, 10),
    "title_size": 12,
    "label_size": 10,
    "tick_size": 9,
}

PARTICIPANTS = ["PDI1", "PDI4"]
PART_SESSIONS = {"PDI1": ["2", "4"], "PDI4": ["2", "3"]}

# Selected (nx, n1) per participant×session for G1 red border
SELECTED_CONFIG = {
    "PDI1_S2": (80, 12),
    "PDI1_S4": (80, 6),
    "PDI4_S2": (80, 10),
    "PDI4_S3": (65, 10),
}

# Grid search parameter ranges (from gs_PDI1_S2_narrow_band.yaml)
NX_VALS = [60, 65, 70, 75, 80]
N1_VALS = [6, 8, 10, 12]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(
    os.environ.get("RESULTS_PATH", _PROJECT_ROOT / "results")
).resolve()

GRID_SEARCH_PARQUET = RESULTS_ROOT / "psid_grid_search_narrow_band" / "results.parquet"


def apply_style() -> None:
    """Set matplotlib rcParams for thesis figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": STYLE["tick_size"],
        "axes.titlesize": STYLE["title_size"],
        "axes.labelsize": STYLE["label_size"],
        "xtick.labelsize": STYLE["tick_size"],
        "ytick.labelsize": STYLE["tick_size"],
        "legend.fontsize": STYLE["tick_size"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_results_for_triplet(triplet) -> List[Dict[str, Any]]:
    """
    Load test split results for a triplet and return list of trial dicts.
    Each dict has keys: Y, Z, stim, participant_id, session, etc.
    """
    res = load_split_results(
        RESULTS_ROOT,
        triplet.psid_variant,
        triplet.psid_run_ts,
        "test",
    )
    if res is None:
        return []

    trials: List[Dict[str, Any]] = []
    # Avoid `x or []` — Y/Z/stim may be numpy arrays (`bool(arr)` is ambiguous).
    def _seq(key: str) -> list:
        v = res.get(key)
        if v is None:
            return []
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return list(v)
        return []

    y_list = _seq("Y")
    z_list = _seq("Z")
    stim_list = _seq("stim")
    pid_list = _seq("participant_id")
    sess_list = _seq("session")

    n = max(len(y_list), len(z_list), len(stim_list))
    for i in range(n):
        y = y_list[i] if i < len(y_list) else None
        z = z_list[i] if i < len(z_list) else None
        stim = stim_list[i] if i < len(stim_list) else None
        pid = pid_list[i] if i < len(pid_list) else None
        sess = sess_list[i] if i < len(sess_list) else None

        # Normalise stim
        if stim is not None:
            s = str(stim).lower()
            stim = "on" if "on" in s else "off"
        else:
            stim = "off"

        trials.append({
            "Y": y,
            "Z": z,
            "stim": stim,
            "participant_id": pid,
            "session": sess,
        })
    return trials


def get_triplets():
    """Return aligned triplets for appendix figures."""
    return _ALL_TRIPLETS
