"""Shared pipeline for all framework pipelines.

``FrameworkPipeline`` drives phases declared in the YAML config.
Phases match experiment config blocks: train, predictions, forecasts, classification.
A phase runs only when its block is present in the YAML.

Config is the raw DotDict loaded from YAML (``get_config()``). Access via
``self.config.experiment``, ``self.config.data``, ``self.config.framework.params``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from training.sweep import run_sweep
from training.test import test
from training.train import train


class FrameworkPipeline:
    """Pipeline driven by YAML config blocks (train/predictions/forecasts/classification)."""

    def __init__(self, config, log, phases: tuple = None, dbs: str = None):
        self.log = log
        self.state: Dict[str, Any] = {"timestamps": {}}
        self.config = config

        self.framework: str = config.framework.name
        self.project_root: Path = Path(config.results.project_root)
        _bool_map = {True: "on", False: "off"}
        self.model_dbs_state: tuple = (
            (dbs,)
            if dbs
            else tuple(
                _bool_map.get(s, s) for s in config.experiment.train.model_dbs_state
            )
        )

        if phases is not None:
            all_known = ("train", "predictions", "forecasts", "classification")
            invalid = [p for p in phases if p not in all_known]
            if invalid:
                raise ValueError(f"unknown phases {invalid}; valid: {all_known}")
            self.phases: tuple = tuple(dict.fromkeys(phases))
        else:
            default_phases = []
            for block in ("train", "predictions", "forecasts", "classification"):
                if getattr(config.experiment, block, None) is not None:
                    default_phases.append(block)
            self.phases = tuple(default_phases)

    # ---------- lifecycle ----------

    def run(self) -> None:
        self.log.info("=" * 60)
        self.log.info(f"{self.framework.upper()} pipeline  phases={list(self.phases)}")
        self.log.info("=" * 60)
        self._run_phase("train", self.phase_train)
        self._run_phase("predictions", self.phase_predictions)
        self._run_phase("forecasts", self.phase_forecasts)
        self._run_phase("classification", self.phase_classification)
        self.log.info("=" * 60)
        self.log.info(f"COMPLETE ({self.framework})")
        self.log.info("=" * 60)

    def _run_phase(self, name: str, fn) -> None:
        if name not in self.phases:
            self.log.info("  SKIP phase: %s", name)
            return
        self.log.info("")
        self.log.info("-- PHASE: %s ---", name)
        fn()

    # ---------- training ----------

    def _train_one(self, model_dbs: str) -> str:
        name = f"{self.config.experiment.name}_dbs_{model_dbs}"
        result_dir = self.project_root / "results" / self.framework / name
        existing = (
            [p for p in result_dir.glob("model_*.pkl") if "_metadata" not in p.name]
            if result_dir.exists()
            else []
        )
        if existing:
            ts = sorted(existing)[-1].stem.replace("model_", "")
            self.log.info("  SKIP training %s - model exists (ts=%s)", name, ts)
            return ts

        self.log.info("  Training %s ...", name)
        train(self.config, model_dbs)
        files = [p for p in result_dir.glob("model_*.pkl") if "_metadata" not in p.name]
        if not files:
            self.log.error("  Training FAILED for %s!", name)
            sys.exit(1)
        return sorted(files)[-1].stem.replace("model_", "")

    def _hydrate_timestamps_from_disk(self) -> None:
        for model_dbs in self.model_dbs_state:
            if self.state["timestamps"].get(model_dbs):
                continue
            name = f"{self.config.experiment.name}_dbs_{model_dbs}"
            result_dir = self.project_root / "results" / self.framework / name
            if not result_dir.exists():
                continue
            files = [
                p for p in result_dir.glob("model_*.pkl") if "_metadata" not in p.name
            ]
            if files:
                ts = sorted(files)[-1].stem.replace("model_", "")
                self.state["timestamps"][model_dbs] = ts
                self.log.info("  hydrated ts[%s]=%s", model_dbs, ts)

    def _require_all_timestamps(self) -> Dict[str, str]:
        if not self.state["timestamps"]:
            self._hydrate_timestamps_from_disk()
        ts = self.state["timestamps"]
        missing = [d for d in self.model_dbs_state if d not in ts]
        if missing:
            raise FileNotFoundError(
                f"missing trained models for dbs={missing}; run --phases train for each first"
            )
        return ts

    def phase_train(self) -> None:
        self.log.info("Train %s on %s", self.framework.upper(), self.model_dbs_state)
        timestamps: dict[str, str] = {}
        for model_dbs in self.model_dbs_state:
            timestamps[model_dbs] = self._train_one(model_dbs)
            self.log.info("  %s complete: ts=%s", model_dbs, timestamps[model_dbs])
        self.state["timestamps"] = timestamps

    # ---------- predictions ----------

    def phase_predictions(self) -> None:
        ts = self._require_all_timestamps()
        splits = list(self.config.experiment.predictions.splits)
        for model_dbs in self.model_dbs_state:
            self.log.info("  predictions  model=%s ts=%s ...", model_dbs, ts[model_dbs])
            test(
                self.config,
                model_dbs=model_dbs,
                run_timestamp=ts[model_dbs],
                incremental=True,
                splits=splits,
            )

    # ---------- forecasts ----------

    def phase_forecasts(self) -> None:
        ts = self._require_all_timestamps()
        splits = list(self.config.experiment.predictions.splits)
        for model_dbs in self.model_dbs_state:
            for h in self.config.experiment.forecasts.h_grid:
                self.log.info(
                    "  forecasts  model=%s h=%.1f ts=%s ...",
                    model_dbs,
                    h,
                    ts[model_dbs],
                )
                test(
                    self.config,
                    model_dbs=model_dbs,
                    run_timestamp=ts[model_dbs],
                    incremental=True,
                    splits=splits,
                    h=h,
                    m=self.config.experiment.forecasts.default_m,
                )

    # ---------- classification ----------

    def phase_classification(self) -> None:
        ts = self._require_all_timestamps()
        cls = self.config.experiment.classification
        fc = self.config.experiment.forecasts
        exp_name = self.config.experiment.name
        variants = {
            model_dbs: f"{exp_name}_dbs_{model_dbs}"
            for model_dbs in self.model_dbs_state
        }
        run_sweep(
            pipeline=self.framework,
            variants=variants,
            timestamps=ts,
            feature_sources_pred=list(cls.feature_sources_pred),
            t_cut_grid=list(cls.t_cut_grid),
            feature_sources_forecast=list(cls.feature_sources_forecast),
            h_grid=list(fc.h_grid),
            m_seconds=fc.default_m,
            m_test_grid=list(fc.m_test_grid),
            classifier_cfg=cls,
            sampling_freq=int(self.config.data.sampling_frequency),
            project_root=self.project_root,
            out_dir=self.project_root
            / "results"
            / self.framework
            / variants["both"]
            / "classification",
            log=self.log,
            config=self.config,
        )
