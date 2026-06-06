"""Shared helpers for thesis notebooks.

Submodules:
    style       — matplotlib style, colors, panel_label
    loaders     — data loading, session discovery, split results
    utils       — trial metrics, normalization helpers
    specs       — figure spec dataclasses
    sec2_common — sec2c/sec2d shared collectors and figure builders
    lib         — legacy figure builders (thesis_lib contents)
"""

from modules.style import (
    apply_thesis_style,
    panel_label,
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    COLOR_PSID,
    COLOR_DPAD,
    COLOR_VARMA,
)
from modules.loaders import (
    EXP_BEHAVIORAL,
    EXP_NEURAL,
    SESSIONS,
    discover_session_run,
    load_split_results_required,
)
from modules.utils import (
    normalize_stim,
    trial_metric_y_for_model,
    trial_metric_z_for_model,
)
