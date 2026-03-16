import pickle
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.classification import (
    load_all_splits,
    prepare_epoched_data,
    run_classification_pipeline,
    _load_framework_for_forecast,
)
from utils.logger import setup_logger, get_logger
from utils.config import get_config
from training.components.tester import Tester


def _get_model_path(project_root: Path, variant_cfg: Any) -> Path:
    return (
        project_root
        / "results"
        / variant_cfg.variant
        / f"model_{variant_cfg.run_ts}.pkl"
    )


def load_psid_model(project_root: Path, variant_cfg: Any) -> Any:
    with open(_get_model_path(project_root, variant_cfg), "rb") as f:
        return pickle.load(f)


def run_classification(
    logger: Any,
    config: Any,
    mode: str,
    feature_source: str,
    history_horizon: Optional[float] = None,
    forecast_horizon: Optional[float] = None,
    variant_dir: Optional[Path] = None,
    run_ts: Optional[str] = None,
    model_on: Optional[Any] = None,
    model_off: Optional[Any] = None,
    model_both: Optional[Any] = None,
) -> None:
    flipped = config.classification.get("flipped", False)
    based_config = dict(
        feature_source=feature_source,
        epoch_length_sec=config.classification.epoch_length,
        overlap=config.classification.epoch_overlap,
        fs=config.classification.sampling_freq,
        n1=config.classification.get("n1"),
        nx=config.classification.get("nx"),
    )

    if variant_dir:
        load_variant_dir = variant_dir
        model_run_ts = run_ts  # Assumes if variant_dir is provided, run_ts (if provided) might be relevant
    elif flipped:
        load_variant_dir = (
            Path(config.results.project_root) / "results" / config.run.dbs_both.variant
        )
        model_run_ts = config.run.dbs_both.run_ts
    else:
        load_variant_dir = (
            Path(config.results.project_root) / "results" / config.run.variant
        )
        model_run_ts = config.run.run_ts

    splits = load_all_splits(load_variant_dir, model_run_ts)
    trainval_trials = [splits.get("train"), splits.get("val")]
    # Filter out None values from trainval_trials
    trainval_trials = [t for t in trainval_trials if t is not None]

    test_trials = [splits.get("test")]
    # Filter out None values from test_trials
    test_trials = [t for t in test_trials if t is not None]

    # Load framework for forecast generation - REQUIRED for forecast mode
    framework = None
    if mode == "forecast" and history_horizon is not None:
        project_root = Path(config.results.project_root)
        framework = _load_framework_for_forecast(
            load_variant_dir, model_run_ts, project_root
        )
        # Framework loading will raise an exception if it fails

    if flipped:
        X_train, y_train, groups_train, _ = prepare_epoched_data(
            trainval_trials,
            **based_config,
            model_on=model_on,
            model_off=model_off,
            model_both=model_both,
            history_horizon=history_horizon,
            forecast_horizon=forecast_horizon,
        )
        X_test, y_test, _, _ = prepare_epoched_data(
            test_trials,
            **based_config,
            target_future=True,
            model_on=model_on,
            model_off=model_off,
            model_both=model_both,
            history_horizon=history_horizon,
            forecast_horizon=forecast_horizon,
        )

        # Log test data availability for flipped case
        if X_test is not None and y_test is not None:
            logger.info(
                f"Test data prepared (flipped): {len(X_test)} samples for {feature_source} {mode}"
            )
        elif len(test_trials) == 0:
            logger.warning(
                f"No test trials available (flipped) for {feature_source} {mode}. "
                f"Test results will not be computed."
            )
        else:
            logger.warning(
                f"Test data preparation returned None (flipped) for {feature_source} {mode}. "
                f"Test trials were provided but no valid data was extracted. "
                f"Test results will not be computed."
            )
    else:
        X_train, y_train, groups_train, _ = prepare_epoched_data(
            trainval_trials,
            **based_config,
            mode=mode,
            history_horizon=history_horizon,
            forecast_horizon=forecast_horizon,
            framework=framework,
        )
        X_test, y_test, _, _ = prepare_epoched_data(
            test_trials,
            **based_config,
            mode=mode,
            history_horizon=history_horizon,
            forecast_horizon=forecast_horizon,
            framework=framework,
        )

    # Log test data availability
    if X_test is not None and y_test is not None:
        logger.info(
            f"Test data prepared: {len(X_test)} samples for {feature_source} {mode}"
        )
    elif len(test_trials) == 0:
        logger.warning(
            f"No test trials available for {feature_source} {mode}. "
            f"Test results will not be computed."
        )
    else:
        logger.warning(
            f"Test data preparation returned None for {feature_source} {mode}. "
            f"Test trials were provided but no valid data was extracted. "
            f"Test results will not be computed."
        )

    results = run_classification_pipeline(
        X_train,
        y_train,
        groups_train,
        X_test,
        y_test,
        config,
        logger,
        feature_source=feature_source,
    )

    # Log whether test_results were computed
    if "test_results" in results:
        logger.info(
            f"Test results computed and saved for {feature_source} {mode} "
            f"(h={history_horizon}, m={forecast_horizon})"
        )
    else:
        logger.warning(
            f"No test results computed for {feature_source} {mode} "
            f"(h={history_horizon}, m={forecast_horizon}). "
            f"Test trials available: {len(test_trials) > 0}"
        )

    results_base = Path(config.results.results_dir)

    if mode == "prediction":
        save_dir = results_base / run_ts
    else:
        save_dir = results_base / run_ts / f"h{history_horizon}_m{forecast_horizon}"

    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"LDA_{feature_source}_{mode}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results, f)


def run_all_classifications(config: Any) -> None:
    logger = get_logger()
    project_root = Path(config.results.project_root)
    flipped = config.classification.get("flipped", False)

    if flipped:
        mode = "flipped"
        model_on = load_psid_model(project_root, config.run.dbs_on)
        model_off = load_psid_model(project_root, config.run.dbs_off)
        model_both = load_psid_model(project_root, config.run.dbs_both)

        history_horizons = config.classification.get("h")
        forecast_horizons = config.classification.get("m")

        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Flipped classification run timestamp: {run_ts}")

        for hh in history_horizons:
            for fh in forecast_horizons:
                logger.info(f"Running flipped suite with h={hh}s, m={fh}s")
                run_classification(
                    logger,
                    config,
                    mode,
                    config.classification.get("prediction_feature_source", "Xp"),
                    history_horizon=hh,
                    forecast_horizon=fh,
                    run_ts=run_ts,
                    model_on=model_on,
                    model_off=model_off,
                    model_both=model_both,
                )
    else:
        verify_test_results(logger, project_root, config)
        history_horizons = config.classification.get("h")
        forecast_horizons = config.classification.get("m")

        logger.info("Running prediction classification (entire trial, h/m ignored)")
        run_classification(
            logger,
            config,
            "prediction",
            config.classification.get("prediction_feature_source", "Xp"),
            history_horizon=None,
            forecast_horizon=None,
            variant_dir=project_root / "results" / config.run.variant,
            run_ts=config.run.run_ts,
        )

        for hh in history_horizons:
            for fh in forecast_horizons:
                logger.info(f"Running forecast classification with h={hh}s, m={fh}s")
                run_classification(
                    logger,
                    config,
                    "forecast",
                    config.classification.get("forecast_feature_source", "Xp"),
                    history_horizon=hh,
                    forecast_horizon=fh,
                    variant_dir=project_root / "results" / config.run.variant,
                    run_ts=config.run.run_ts,
                )


def verify_test_results(logger: Any, project_root: Path, config: Any) -> None:
    save_ts = config.run.run_ts
    variant_results_dir = project_root / "results" / config.run.variant
    test_results_path = variant_results_dir / "test" / f"test_results_{save_ts}.parquet"

    if not test_results_path.exists():
        setup_paths = list(
            (project_root / "training" / "setups").rglob(f"{config.run.variant}.yaml")
        )
        if not setup_paths:
            raise FileNotFoundError(
                f"Could not find training setup for variant: {config.run.variant}"
            )
        tester = Tester(
            get_config(str(setup_paths[0])),
            run_timestamp=config.run.run_ts,
        )
        tester.run_predictions()
        tester.save_results()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified DBS Classification")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = get_config(args.config)
    setup_logger(config.results.log_dir, name=__file__)
    run_all_classifications(config)


if __name__ == "__main__":
    main()
