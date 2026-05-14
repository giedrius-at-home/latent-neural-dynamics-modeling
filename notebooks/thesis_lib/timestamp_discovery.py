"""Discover newest ``test`` run timestamps on disk vs values baked into ``specs.py``."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from utils.thesis_result_timestamps import latest_run_timestamp_on_disk

from dashboard.thesis.errors import ThesisDataError
from dashboard.thesis.specs import (
    AlignedTriplet,
    ClassificationF1PickleRef,
    LatentPhasePanel,
    LatentPhaseRow,
    PsidCyPanel,
    PsidCyRow,
    StripPanelEntry,
    ThesisAggregateRmseSpec,
    ThesisC2ForecastSpec,
    ThesisClassificationF1Spec,
    ThesisCrossBlockSpec,
    ThesisForecastCheckpointSpec,
    ThesisFigureSpec,
    ThesisForecastRmseSpec,
    ThesisLatentPhaseSpec,
    ThesisNeuralBandHeatmapSpec,
    ThesisNeuralTimeseriesSpec,
    ThesisPsidCyImportanceSpec,
    ThesisPsidCzSpec,
    ThesisStripPanelsSpec,
    ThesisWithinCrossSpec,
    THESIS_AGGREGATE_FIGURES,
    THESIS_C2_FORECASTS,
    THESIS_CLASSIFICATION_F1,
    THESIS_CROSS_BLOCK,
    THESIS_FORECAST_CHECKPOINT,
    THESIS_FIGURES,
    THESIS_FORECAST_FIGURES,
    THESIS_LATENT_PHASE,
    THESIS_NEURAL_BAND_HEATMAPS,
    THESIS_NEURAL_FORECAST_FIGURES,
    THESIS_NEURAL_TIMESERIES,
    THESIS_PSID_CY_IMPORTANCE,
    THESIS_PSID_CZ_HEATMAP,
    THESIS_STRIP_PANELS,
    THESIS_WITHIN_CROSS,
    _ALL_TRIPLETS,
    _variant_cross_eval,
    _variant_off,
    _variant_on,
)


def iter_spec_variant_run_ts(tri: AlignedTriplet) -> Iterable[tuple[str, str, str]]:
    """Yield ``(results_subdir, run_ts, role)`` for every thesis result folder implied by *tri*."""
    yield (tri.psid_variant, tri.psid_run_ts, f"{tri.label}|psid_joint")
    yield (
        _variant_off(tri.psid_variant),
        tri.psid_run_ts_off or "",
        f"{tri.label}|psid_off",
    )
    yield (
        _variant_on(tri.psid_variant),
        tri.psid_run_ts_on or "",
        f"{tri.label}|psid_on",
    )

    yield (tri.dpad_variant, tri.dpad_run_ts, f"{tri.label}|dpad_joint")
    yield (
        _variant_off(tri.dpad_variant),
        tri.dpad_run_ts_off or "",
        f"{tri.label}|dpad_off",
    )
    yield (
        _variant_on(tri.dpad_variant),
        tri.dpad_run_ts_on or "",
        f"{tri.label}|dpad_on",
    )

    yield (tri.varma_variant, tri.varma_run_ts, f"{tri.label}|varma_joint")
    yield (
        _variant_off(tri.varma_variant),
        tri.varma_run_ts_off or "",
        f"{tri.label}|varma_off",
    )
    yield (
        _variant_on(tri.varma_variant),
        tri.varma_run_ts_on or "",
        f"{tri.label}|varma_on",
    )

    yield (
        _variant_cross_eval(_variant_on(tri.varma_variant), "off"),
        tri.varma_run_ts_eval_off or "",
        f"{tri.label}|varma_eval_off",
    )
    yield (
        _variant_cross_eval(_variant_off(tri.varma_variant), "on"),
        tri.varma_run_ts_eval_on or "",
        f"{tri.label}|varma_eval_on",
    )


def spec_timestamps_by_variant(triplets: list[AlignedTriplet]) -> dict[str, set[str]]:
    """Map each ``results/<variant>/`` name to the set of spec run timestamps that reference it."""
    by_var: dict[str, set[str]] = defaultdict(set)
    for tri in triplets:
        for variant, run_ts, _role in iter_spec_variant_run_ts(tri):
            if not run_ts.strip():
                continue
            by_var[variant].add(run_ts.strip())
    return dict(by_var)


def compare_spec_to_disk(
    results_root: Path,
    *,
    split: str = "test",
    triplets: list[AlignedTriplet] | None = None,
) -> list[dict[str, str | bool | None]]:
    """
    For each variant directory referenced in *triplets*, compare spec timestamp(s) to
    ``latest_run_timestamp_on_disk``. Returns one row per variant (not per triplet role).
    """
    trips = triplets if triplets is not None else list(_ALL_TRIPLETS)
    by_var = spec_timestamps_by_variant(trips)
    rows: list[dict[str, str | bool | None]] = []
    for variant in sorted(by_var.keys()):
        spec_set = by_var[variant]
        spec_ts = max(spec_set) if len(spec_set) == 1 else ""
        multi = len(spec_set) > 1
        variant_dir = results_root / variant
        latest, src = latest_run_timestamp_on_disk(variant_dir, split=split)
        stale = bool(latest and spec_ts and latest != spec_ts)
        rows.append(
            {
                "variant": variant,
                "spec_run_ts": (
                    spec_ts if not multi else "MULTIPLE: " + ",".join(sorted(spec_set))
                ),
                "latest_on_disk": latest,
                "disk_source": src,
                "spec_conflict": multi,
                "stale_vs_disk": stale,
                "newer_on_disk": bool(
                    latest and spec_ts and not multi and latest > spec_ts
                ),
            }
        )
    return rows


def _require_latest_ts(variant_dir: Path, *, split: str) -> str:
    ts, _src = latest_run_timestamp_on_disk(variant_dir, split=split)
    if ts is None:
        raise ThesisDataError(
            f"No thesis results found under {variant_dir} (split subdirectory={split!r})."
        )
    return ts


def _optional_latest_ts(variant_dir: Path, *, split: str) -> str:
    """Like ``_require_latest_ts`` but returns empty string when no loadable results exist.

    Use this for frameworks (e.g. DPAD while retraining) whose absence should degrade
    gracefully. Only returns a timestamp when the corresponding split parquet (or nested
    pickle directory) is present — a bare ``model_*.pkl`` without a ``test_results_*.parquet``
    would lead to a silent ``None`` load, which downstream code would then crash on.
    """
    ts, src = latest_run_timestamp_on_disk(variant_dir, split=split)
    if not ts:
        return ""
    if src == "model_pkl":
        pq = variant_dir / split / f"test_results_{ts}.parquet"
        pkl_dir = variant_dir / ts
        if not pq.exists() and not pkl_dir.is_dir():
            return ""
    return ts


def resolve_aligned_triplet_timestamps(
    tri: AlignedTriplet,
    results_root: Path,
    *,
    split: str = "test",
) -> AlignedTriplet:
    """Fill *tri* with newest on-disk run id per implied ``results/<variant>/`` folder.

    DPAD timestamps are optional: if the DPAD variant directory is absent (e.g. retraining),
    the corresponding fields fall back to empty string so loaders can skip DPAD gracefully.
    """
    rr = results_root

    def req(rel: str) -> str:
        return _require_latest_ts(rr / rel, split=split)

    def opt(rel: str) -> str:
        return _optional_latest_ts(rr / rel, split=split)

    return replace(
        tri,
        psid_run_ts=req(tri.psid_variant),
        psid_run_ts_off=req(_variant_off(tri.psid_variant)),
        psid_run_ts_on=req(_variant_on(tri.psid_variant)),
        dpad_run_ts=opt(tri.dpad_variant),
        dpad_run_ts_off=opt(_variant_off(tri.dpad_variant)),
        dpad_run_ts_on=opt(_variant_on(tri.dpad_variant)),
        varma_run_ts=req(tri.varma_variant),
        varma_run_ts_off=req(_variant_off(tri.varma_variant)),
        varma_run_ts_on=req(_variant_on(tri.varma_variant)),
        varma_run_ts_eval_off=req(
            _variant_cross_eval(_variant_on(tri.varma_variant), "off")
        ),
        varma_run_ts_eval_on=req(
            _variant_cross_eval(_variant_off(tri.varma_variant), "on")
        ),
    )


def resolve_joint_framework_timestamps(
    spec: ThesisFigureSpec | ThesisC2ForecastSpec | ThesisNeuralTimeseriesSpec,
    results_root: Path,
    *,
    split: str = "test",
) -> ThesisFigureSpec | ThesisC2ForecastSpec | ThesisNeuralTimeseriesSpec:
    """Resolve PSID / DPAD / VARMA joint (and optional VARMA OFF/ON) timestamps to latest on disk.

    DPAD timestamp is optional: when the DPAD variant directory is absent (e.g. retraining),
    ``dpad_run_ts`` falls back to empty string and downstream loaders skip DPAD gracefully.
    """
    rr = results_root
    kw = dict(
        psid_run_ts=_require_latest_ts(rr / spec.psid_variant, split=split),
        dpad_run_ts=_optional_latest_ts(rr / spec.dpad_variant, split=split),
        varma_run_ts=_require_latest_ts(rr / spec.varma_variant, split=split),
    )
    # dbs_both joint runs need OFF/ON parquets; always take latest there (do not rely on
    # infer_varma_off_on_run_ts, which keys off the pre-resolve joint timestamp in specs._ALL_TRIPLETS).
    if "dbs_both" in spec.varma_variant:
        kw["varma_run_ts_off"] = _require_latest_ts(
            rr / _variant_off(spec.varma_variant), split=split
        )
        kw["varma_run_ts_on"] = _require_latest_ts(
            rr / _variant_on(spec.varma_variant), split=split
        )
    else:
        if spec.varma_run_ts_off is not None:
            kw["varma_run_ts_off"] = _require_latest_ts(
                rr / _variant_off(spec.varma_variant), split=split
            )
        if spec.varma_run_ts_on is not None:
            kw["varma_run_ts_on"] = _require_latest_ts(
                rr / _variant_on(spec.varma_variant), split=split
            )
    return replace(spec, **kw)


def _resolve_classification_pickle_relative_path(
    rel: str,
    results_root: Path,
    classification_parent: str,
    *,
    split: str,
) -> str:
    parts = Path(rel).parts
    if len(parts) < 3:
        return rel
    variant = parts[0]
    tail = Path(*parts[2:])
    base = results_root / classification_parent / variant
    ts = _require_latest_ts(base, split=split)
    return str(Path(variant) / ts / tail)


def _resolve_classification_f1_spec(
    spec: ThesisClassificationF1Spec,
    results_root: Path,
    *,
    split: str,
) -> ThesisClassificationF1Spec:
    parent = spec.classification_parent
    new_points = tuple(
        replace(
            p,
            pickle_relative_path=_resolve_classification_pickle_relative_path(
                p.pickle_relative_path, results_root, parent, split=split
            ),
        )
        for p in spec.points
    )
    return replace(spec, points=new_points)


def _resolve_psid_cy_panel(
    p: PsidCyPanel, results_root: Path, *, split: str
) -> PsidCyPanel:
    ts = _require_latest_ts(results_root / p.psid_variant, split=split)
    return replace(p, psid_run_ts=ts)


def _resolve_psid_cy_spec(
    spec: ThesisPsidCyImportanceSpec, results_root: Path, *, split: str
) -> ThesisPsidCyImportanceSpec:
    new_rows: list[PsidCyRow] = []
    for row in spec.rows:
        new_panels = tuple(
            _resolve_psid_cy_panel(p, results_root, split=split) for p in row.panels
        )
        new_rows.append(replace(row, panels=new_panels))
    return replace(spec, rows=tuple(new_rows))


def _resolve_psid_cz_spec(
    spec: ThesisPsidCzSpec, results_root: Path, *, split: str
) -> ThesisPsidCzSpec:
    new_rows: list[PsidCyRow] = []
    for row in spec.rows:
        new_panels = tuple(
            _resolve_psid_cy_panel(p, results_root, split=split) for p in row.panels
        )
        new_rows.append(replace(row, panels=new_panels))
    return replace(spec, rows=tuple(new_rows))


def _resolve_latent_phase_panel(
    p: LatentPhasePanel, results_root: Path, *, split: str
) -> LatentPhasePanel:
    return replace(
        p,
        psid_run_ts=_require_latest_ts(results_root / p.psid_variant, split=split),
        dpad_run_ts=_optional_latest_ts(results_root / p.dpad_variant, split=split),
    )


def _resolve_latent_phase_spec(
    spec: ThesisLatentPhaseSpec, results_root: Path, *, split: str
) -> ThesisLatentPhaseSpec:
    new_rows: list[LatentPhaseRow] = []
    for row in spec.rows:
        new_panels = tuple(
            _resolve_latent_phase_panel(p, results_root, split=split)
            for p in row.panels
        )
        new_rows.append(replace(row, panels=new_panels))
    return replace(spec, rows=tuple(new_rows))


def _resolve_strip_panels_spec(
    spec: ThesisStripPanelsSpec, results_root: Path, *, split: str
) -> ThesisStripPanelsSpec:
    new_entries: list[StripPanelEntry] = [
        replace(
            e,
            triplet=resolve_aligned_triplet_timestamps(
                e.triplet, results_root, split=split
            ),
        )
        for e in spec.panels
    ]
    return replace(spec, panels=new_entries)


@dataclass(frozen=True)
class ThesisDashboardSpecs:
    """Thesis figure specs with run timestamps resolved to latest on-disk checkpoints/parquets."""

    figures: tuple[ThesisFigureSpec, ...]
    neural_timeseries: tuple[ThesisNeuralTimeseriesSpec, ...]
    c2_forecasts: tuple[ThesisC2ForecastSpec, ...]
    cross_block: tuple[ThesisCrossBlockSpec, ...]
    forecast_checkpoint: tuple[ThesisForecastCheckpointSpec, ...]
    neural_forecast_figures: tuple[ThesisForecastRmseSpec, ...]
    classification_f1: tuple[ThesisClassificationF1Spec, ...]
    aggregate_figures: tuple[ThesisAggregateRmseSpec, ...]
    forecast_figures: tuple[ThesisForecastRmseSpec, ...]
    neural_band_heatmaps: tuple[ThesisNeuralBandHeatmapSpec, ...]
    latent_phase: tuple[ThesisLatentPhaseSpec, ...]
    psid_cy_importance: tuple[ThesisPsidCyImportanceSpec, ...]
    psid_cz_heatmap: tuple[ThesisPsidCzSpec, ...]
    strip_panels: tuple[ThesisStripPanelsSpec, ...]
    within_cross: tuple[ThesisWithinCrossSpec, ...]


def build_thesis_dashboard_specs(
    results_root: Path,
    *,
    split: str = "test",
) -> ThesisDashboardSpecs:
    """Resolve every *AlignedTriplet* / framework timestamp in the default thesis lists to newest on disk."""
    rr = results_root.resolve()

    def rt(tri: AlignedTriplet) -> AlignedTriplet:
        return resolve_aligned_triplet_timestamps(tri, rr, split=split)

    def rjf(
        spec: ThesisFigureSpec | ThesisC2ForecastSpec | ThesisNeuralTimeseriesSpec,
    ) -> ThesisFigureSpec | ThesisC2ForecastSpec | ThesisNeuralTimeseriesSpec:
        return resolve_joint_framework_timestamps(spec, rr, split=split)

    return ThesisDashboardSpecs(
        figures=tuple(rjf(s) for s in THESIS_FIGURES),
        neural_timeseries=tuple(rjf(s) for s in THESIS_NEURAL_TIMESERIES),
        c2_forecasts=tuple(rjf(s) for s in THESIS_C2_FORECASTS),
        cross_block=tuple(
            replace(s, joint_triplet=rt(s.joint_triplet)) for s in THESIS_CROSS_BLOCK
        ),
        forecast_checkpoint=tuple(
            replace(s, joint_triplet=rt(s.joint_triplet))
            for s in THESIS_FORECAST_CHECKPOINT
        ),
        neural_forecast_figures=tuple(
            replace(s, triplets=tuple(rt(t) for t in s.triplets))
            for s in THESIS_NEURAL_FORECAST_FIGURES
        ),
        classification_f1=tuple(
            _resolve_classification_f1_spec(s, rr, split=split)
            for s in THESIS_CLASSIFICATION_F1
        ),
        aggregate_figures=tuple(
            replace(s, triplets=tuple(rt(t) for t in s.triplets))
            for s in THESIS_AGGREGATE_FIGURES
        ),
        forecast_figures=tuple(
            replace(s, triplets=tuple(rt(t) for t in s.triplets))
            for s in THESIS_FORECAST_FIGURES
        ),
        neural_band_heatmaps=tuple(
            replace(s, triplets=tuple(rt(t) for t in s.triplets))
            for s in THESIS_NEURAL_BAND_HEATMAPS
        ),
        latent_phase=tuple(
            _resolve_latent_phase_spec(s, rr, split=split) for s in THESIS_LATENT_PHASE
        ),
        psid_cy_importance=tuple(
            _resolve_psid_cy_spec(s, rr, split=split) for s in THESIS_PSID_CY_IMPORTANCE
        ),
        psid_cz_heatmap=tuple(
            _resolve_psid_cz_spec(s, rr, split=split) for s in THESIS_PSID_CZ_HEATMAP
        ),
        strip_panels=tuple(
            _resolve_strip_panels_spec(s, rr, split=split) for s in THESIS_STRIP_PANELS
        ),
        within_cross=tuple(
            replace(wc, joint_triplet=rt(wc.joint_triplet))
            for wc in THESIS_WITHIN_CROSS
        ),
    )
