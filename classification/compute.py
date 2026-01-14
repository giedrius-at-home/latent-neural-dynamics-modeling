import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.classification import (
    CLASSIFIERS,
    prepare_epoched_data,
    run_grid_search_cv,
    evaluate_on_test_set,
)
from utils.logger import setup_logger, get_logger
from utils.config import get_config


def load_all_splits(
    variant_dir: Path, run_ts: str
) -> Dict[str, Optional[Dict[str, Any]]]:
    from dashboard.subtabs import load_precomputed_results

    splits = {}
    for split_name in ["train", "val", "test"]:
        splits[split_name] = load_precomputed_results(variant_dir, run_ts, split_name)
    return splits


def save_classification_results(results: Dict[str, Any], save_path: Path, logger):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved results to {save_path}")


def compute_classification_for_config(
    variant_dir: Path,
    run_ts: str,
    mode: str,
    feature_source: str,
    epoch_length_sec: float,
    epoch_overlap: float,
    n_splits: int,
    logger,
    forecast_horizon_sec: Optional[float] = None,
):
    logger.info("=" * 80)
    logger.info(f"Computing classification for:")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Feature Source: {feature_source}")
    logger.info(f"  Epoch Length: {epoch_length_sec}s")
    logger.info(f"  Epoch Overlap: {epoch_overlap}")
    logger.info(f"  CV Folds: {n_splits}")
    if forecast_horizon_sec:
        logger.info(f"  Forecast Horizon: {forecast_horizon_sec}s")
    logger.info("=" * 80)

    all_splits = load_all_splits(variant_dir, run_ts)

    train_res = all_splits.get("train")
    val_res = all_splits.get("val")
    test_res = all_splits.get("test")

    trainval_list = [r for r in [train_res, val_res] if r is not None]
    test_list = [test_res] if test_res is not None else []

    if not trainval_list:
        logger.error("No train or val results found. Skipping.")
        return

    prep_kwargs = {
        "feature_source": feature_source,
        "epoch_length_sec": epoch_length_sec,
        "overlap": epoch_overlap,
    }

    if forecast_horizon_sec is not None:
        prep_kwargs["forecast_horizon_sec"] = forecast_horizon_sec

    logger.info("Extracting epoched features...")
    X_trainval, y_trainval, groups_trainval, meta_trainval = prepare_epoched_data(
        trainval_list, **prep_kwargs
    )
    X_test, y_test, groups_test, meta_test = (
        prepare_epoched_data(test_list, **prep_kwargs)
        if test_list
        else (None, None, None, None)
    )

    if X_trainval is None or len(X_trainval) == 0:
        logger.error("No valid epochs found for classification. Skipping.")
        return

    logger.info(f"Train+Val: {len(X_trainval)} epochs, {X_trainval.shape[1]} features")
    logger.info(
        f"  DBS ON: {np.sum(y_trainval == 1)} | DBS OFF: {np.sum(y_trainval == 0)}"
    )

    if X_test is not None and len(X_test) > 0:
        logger.info(f"Test: {len(X_test)} epochs, {X_test.shape[1]} features")
        logger.info(f"  DBS ON: {np.sum(y_test == 1)} | DBS OFF: {np.sum(y_test == 0)}")
    else:
        logger.info("Test: No test data available")

    results_dir = Path(variant_dir) / run_ts / "classification"
    results_dir.mkdir(parents=True, exist_ok=True)

    epoch_params_str = f"epoch{epoch_length_sec}s_overlap{epoch_overlap}"

    for clf_name in CLASSIFIERS:
        logger.info(f"\n--- Running {clf_name} ---")

        key_base = f"{clf_name}_{feature_source}_{mode}_{epoch_params_str}".replace(
            " ", "_"
        ).replace(".", "p")
        if forecast_horizon_sec:
            key_base += f"_fh{forecast_horizon_sec}s".replace(".", "p")

        cache_path = results_dir / f"{key_base}.pkl"

        if cache_path.exists():
            logger.info(f"Results already exist at {cache_path}, skipping...")
            continue

        logger.info("Running grid search with ChronoGroupsSplit CV...")
        best_params, best_score, cv_results = run_grid_search_cv(
            clf_name, X_trainval, y_trainval, groups_trainval, n_splits
        )

        logger.info(f"Best CV Score: {best_score:.4f}")
        logger.info(f"Best Params: {best_params}")

        if X_test is not None and y_test is not None and len(y_test) > 0:
            best_pipeline = cv_results.get("best_pipeline")
            if best_pipeline is not None:
                logger.info("Evaluating on test set...")
                test_results = evaluate_on_test_set(best_pipeline, X_test, y_test)
                cv_results["test_results"] = test_results
                logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
                logger.info(
                    f"Test Balanced Accuracy: {test_results['balanced_accuracy']:.4f}"
                )

        save_classification_results(cv_results, cache_path, logger)


def compute(config):
    logger = get_logger()
    logger.info("Starting DBS classification computation...")

    project_root = Path(config.results.project_root)
    results_root = project_root / "results"
    variant_dir = results_root / config.run.variant

    if not variant_dir.exists():
        logger.error(f"Variant directory not found: {variant_dir}")
        sys.exit(1)

    run_ts = config.run.run_ts

    train_dir = variant_dir / f"train_results_{run_ts}"
    val_dir = variant_dir / f"val_results_{run_ts}"
    test_dir = variant_dir / f"test_results_{run_ts}"

    if not train_dir.exists():
        logger.error(f"Train results not found: {train_dir}")
        sys.exit(1)

    logger.info(f"Variant: {config.run.variant}")
    logger.info(f"Run Timestamp: {run_ts}")
    logger.info(f"Train results: {train_dir} (exists)")
    logger.info(
        f"Val results: {val_dir} ({'exists' if val_dir.exists() else 'not found'})"
    )
    logger.info(
        f"Test results: {test_dir} ({'exists' if test_dir.exists() else 'not found'})"
    )

    if not test_dir.exists():
        logger.warning("Test predictions not found. Generating test predictions...")
        logger.info(
            "This is similar to running: python training/test.py --config <config> --run <run_ts>"
        )

        from training.components.tester import Tester
        from utils.config import get_config as load_config

        training_config_path = (
            project_root / "training" / "setups" / f"{config.run.variant}.yaml"
        )
        if not training_config_path.exists():
            logger.error(f"Training config not found: {training_config_path}")
            logger.error("Cannot generate test predictions without training config.")
            logger.info(
                "Please run: python training/test.py --config <training_config> --run <run_ts>"
            )
            sys.exit(1)

        training_config = load_config(str(training_config_path))
        tester = Tester(training_config, run_timestamp=run_ts)
        tester.run_predictions()
        tester.save_results()
        logger.info("Test predictions generated and saved.")

    logger.info(f"CV Folds: {config.classification.n_splits}")
    logger.info(f"Epoch Length: {config.classification.epoch_length}s")
    logger.info(f"Epoch Overlap: {config.classification.epoch_overlap}")

    logger.info("\n" + "=" * 80)
    logger.info("COMPUTING CLASSIFICATION FROM PREDICTIONS")
    logger.info("Using saved predictions from training (train/val) and test")
    logger.info("=" * 80)

    compute_classification_for_config(
        variant_dir=variant_dir,
        run_ts=run_ts,
        mode="prediction",
        feature_source=config.classification.prediction_feature_source,
        epoch_length_sec=config.classification.epoch_length,
        epoch_overlap=config.classification.epoch_overlap,
        n_splits=config.classification.n_splits,
        logger=logger,
    )

    logger.info("\n" + "=" * 80)
    logger.info("COMPUTING CLASSIFICATION FROM FORECASTS")
    logger.info("Using saved forecasts from training (train/val) and test")
    logger.info(
        f"Forecast parameters from training config: model.forecast.m and model.forecast.history"
    )
    logger.info("=" * 80)

    compute_classification_for_config(
        variant_dir=variant_dir,
        run_ts=run_ts,
        mode="forecast",
        feature_source=config.classification.forecast_feature_source,
        epoch_length_sec=config.classification.epoch_length,
        epoch_overlap=config.classification.epoch_overlap,
        n_splits=config.classification.n_splits,
        logger=logger,
    )

    logger.info("\n" + "=" * 80)
    logger.info("DONE! All classification results computed and saved.")
    logger.info("=" * 80)


def main(args):
    config = get_config(args.config)
    logger = setup_logger(config.results.log_dir, name=__file__)

    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Config content:\n{config}")

    compute(config)

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute DBS classification using specified configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    main(args)
