"""Discover newest run timestamps under ``results/<variant>/`` (no torch or dashboard imports)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = ["latest_test_parquet_ts", "latest_run_timestamp_on_disk"]


def latest_timestamp_subdir_with_pkls(variant_dir: Path) -> Optional[str]:
    """
    Newest immediate subdirectory name under *variant_dir* that contains at least one ``*.pkl``.

    Used for ``results/classification/<variant>/<run_ts>/*.pkl`` layouts (no top-level ``model_*.pkl``).
    """
    if not variant_dir.is_dir():
        return None
    names: list[str] = []
    for child in variant_dir.iterdir():
        if child.is_dir() and any(child.glob("*.pkl")):
            names.append(child.name)
    return max(names) if names else None


def latest_test_parquet_ts(split_dir: Path) -> Optional[str]:
    """Newest timestamp stem from ``test_results_<ts>.parquet`` files in *split_dir*."""
    if not split_dir.is_dir():
        return None
    candidates: list[str] = []
    for child in split_dir.iterdir():
        name = child.name
        if name.startswith("test_results_") and name.endswith(".parquet"):
            ts = name.replace("test_results_", "").replace(".parquet", "")
            candidates.append(ts)
    return max(candidates) if candidates else None


def latest_run_timestamp_on_disk(
    variant_dir: Path, *, split: str = "test"
) -> tuple[Optional[str], str]:
    """
    Newest run id among:

    - top-level ``*.pkl`` (via ``get_latest_timestamp``),
    - ``{split}/test_results_<ts>.parquet``,
    - timestamp-named subdirs containing ``*.pkl`` (classification-style).

    Returns ``(ts | None, source)`` where *source* is ``model_pkl``, ``test_parquet``,
    ``nested_pkl``, ``none``, or ``mixed`` (multiple sources tie for max).
    """
    ts_pkl: Optional[str] = None
    if variant_dir.is_dir():
        try:
            from utils.miscellaneous import get_latest_timestamp

            ts_pkl = get_latest_timestamp(str(variant_dir))
        except (ValueError, StopIteration, OSError):
            ts_pkl = None

    ts_pq = latest_test_parquet_ts(variant_dir / split)
    ts_nested = latest_timestamp_subdir_with_pkls(variant_dir)

    candidates: list[tuple[str, str]] = []
    if ts_pkl is not None:
        candidates.append((ts_pkl, "model_pkl"))
    if ts_pq is not None:
        candidates.append((ts_pq, "test_parquet"))
    if ts_nested is not None:
        candidates.append((ts_nested, "nested_pkl"))
    if not candidates:
        return None, "none"
    best_ts = max(t for t, _ in candidates)
    sources = sorted({s for t, s in candidates if t == best_ts})
    if len(sources) == 1:
        return best_ts, sources[0]
    return best_ts, "mixed"
