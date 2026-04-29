import copy
import os
import pickle
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime


sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.classification import (
    apply_lda_permutation_to_results,
    load_all_splits,
    prepare_epoched_data,
    run_classification_pipeline,
    run_lda_fit_and_test_only,
    _load_framework_for_forecast,
)
from utils.classification.cache import (
    build_flipped_epoched_key,
    build_lda_fit_key,
    build_standard_epoched_key,
    key_hash,
    save_flipped_epoched,
    save_lda_fit,
    save_standard_epoched,
    try_load_flipped_epoched,
    try_load_lda_fit,
    try_load_standard_epoched,
)
from utils.logger import setup_logger, get_logger
from utils.config import get_config
from training.components.tester import Tester

# In-process LDA fit cache (grid+test only) to avoid duplicate work within one Python run.
_RUNTIME_LDA_FIT_CACHE: Dict[str, Dict[str, Any]] = {}

# Skip repeated identical jobs in a single python invocation (same config file).
_SEEN_INTRARUN_KEYS: Set[Tuple[Any, ...]] = set()


def _intrarun_dedup_enabled() -> bool:
    return os.environ.get("CLASSIFICATION_INTRARUN_DEDUP", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _norm_hm(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(x)


def _intrarun_key(
    config: Any,
    flipped: bool,
    mode: str,
    feature_source: str,
    history_horizon: Optional[float],
    forecast_horizon: Optional[float],
) -> Tuple[Any, ...]:
    run_id = str(config.run.dbs_both.variant) if flipped else str(config.run.variant)
    return (
        run_id,
        flipped,
        mode,
        feature_source,
        _norm_hm(history_horizon),
        _norm_hm(forecast_horizon),
    )


def _unique_h_m_pairs(
    history_horizons: List[Any], forecast_horizons: List[Any]
) -> List[Tuple[float, float]]:
    seen: Set[Tuple[float, float]] = set()
    out: List[Tuple[float, float]] = []
    for hh in history_horizons:
        for fh in forecast_horizons:
            pair = (float(hh), float(fh))
            if pair not in seen:
                seen.add(pair)
                out.append(pair)
    return out


def _get_model_path(project_root: Path, variant_cfg: Any) -> Path:
    return (
        project_root
        / "results"
        / variant_cfg.variant
        / f"model_{variant_cfg.run_ts}.pkl"
    )


def load_psid_model(project_root: Path, variant_cfg: Any) -> Any:
    with open(_get_model_path(project_root, variant_cfg), "rb") as f:
        model = pickle.load(f)
    if hasattr(model, "restoreModels"):
        model.restoreModels()
    return model


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
    if _intrarun_dedup_enabled():
        ik = _intrarun_key(
            config, flipped, mode, feature_source, history_horizon, forecast_horizon
        )
        if ik in _SEEN_INTRARUN_KEYS:
            logger.info(
                f"Skipping duplicate classification within this process (same as an earlier call in this run): "
                f"mode={mode} feature={feature_source} h={history_horizon} m={forecast_horizon} flipped={flipped}"
            )
            return
        _SEEN_INTRARUN_KEYS.add(ik)

    based_config = dict(
        feature_source=feature_source,
        epoch_length_sec=config.classification.epoch_length,
        overlap=config.classification.epoch_overlap,
        fs=config.classification.sampling_freq,
        n1=config.classification.get("n1"),
        nx=config.classification.get("nx"),
    )
    permutation_test = config.classification.get("permutation_test", False)
    project_root = Path(config.results.project_root)

    if variant_dir:
        load_variant_dir = variant_dir
        model_run_ts = run_ts
    elif flipped:
        load_variant_dir = project_root / "results" / config.run.dbs_both.variant
        model_run_ts = config.run.dbs_both.run_ts
    else:
        load_variant_dir = project_root / "results" / config.run.variant
        model_run_ts = config.run.run_ts

    splits = load_all_splits(load_variant_dir, model_run_ts)
    trainval_trials = [splits.get("train"), splits.get("val")]
    trainval_trials = [t for t in trainval_trials if t is not None]

    test_trials = [splits.get("test")]
    test_trials = [t for t in test_trials if t is not None]

    framework = None
    if mode == "forecast" and history_horizon is not None:
        framework = _load_framework_for_forecast(
            load_variant_dir, model_run_ts, project_root
        )

    epoched_cache_key: Optional[Dict[str, Any]] = None
    epoched_cache_hash: Optional[str] = None

    if flipped:
        if (
            mode == "flipped"
            and history_horizon is not None
            and forecast_horizon is not None
        ):
            epoched_cache_key = build_flipped_epoched_key(
                project_root,
                config,
                feature_source,
                history_horizon,
                forecast_horizon,
                load_variant_dir,
                model_run_ts,
            )
            epoched_cache_hash = key_hash(epoched_cache_key)
            cached = try_load_flipped_epoched(
                project_root, epoched_cache_key, epoched_cache_hash
            )
            if cached is not None:
                X_train, y_train, groups_train, X_test, y_test, groups_test = cached
                logger.info(
                    f"Flipped epoched arrays loaded from disk cache ({epoched_cache_hash[:20]}...)"
                )
            else:
                X_train, y_train, groups_train, _ = prepare_epoched_data(
                    trainval_trials,
                    **based_config,
                    model_on=model_on,
                    model_off=model_off,
                    model_both=model_both,
                    history_horizon=history_horizon,
                    forecast_horizon=forecast_horizon,
                )
                X_test, y_test, groups_test, _ = prepare_epoched_data(
                    test_trials,
                    **based_config,
                    target_future=True,
                    model_on=model_on,
                    model_off=model_off,
                    model_both=model_both,
                    history_horizon=history_horizon,
                    forecast_horizon=forecast_horizon,
                )
                if X_train is not None:
                    save_flipped_epoched(
                        project_root,
                        epoched_cache_key,
                        epoched_cache_hash,
                        X_train,
                        y_train,
                        groups_train,
                        X_test,
                        y_test,
                        groups_test,
                    )
                    logger.info(
                        f"Wrote flipped epoched disk cache ({epoched_cache_hash[:20]}... under results/classification_cache/flipped)"
                    )
        else:
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
        epoched_cache_key = build_standard_epoched_key(
            project_root,
            config,
            feature_source,
            mode,
            history_horizon,
            forecast_horizon,
            load_variant_dir,
            model_run_ts,
        )
        epoched_cache_hash = key_hash(epoched_cache_key)
        cached_std = try_load_standard_epoched(
            project_root, epoched_cache_key, epoched_cache_hash
        )
        if cached_std is not None:
            X_train, y_train, groups_train, X_test, y_test, groups_test = cached_std
            logger.info(
                f"Standard epoched arrays loaded from disk cache ({epoched_cache_hash[:20]}...)"
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
            X_test, y_test, groups_test, _ = prepare_epoched_data(
                test_trials,
                **based_config,
                mode=mode,
                history_horizon=history_horizon,
                forecast_horizon=forecast_horizon,
                framework=framework,
            )
            if X_train is not None:
                save_standard_epoched(
                    project_root,
                    epoched_cache_key,
                    epoched_cache_hash,
                    X_train,
                    y_train,
                    groups_train,
                    X_test,
                    y_test,
                    groups_test,
                )
                logger.info(
                    f"Wrote standard epoched disk cache ({epoched_cache_hash[:20]}... under results/classification_cache/epoched)"
                )

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

    results: Optional[Dict[str, Any]] = None
    if epoched_cache_key is not None and epoched_cache_hash is not None:
        lda_key = build_lda_fit_key(
            epoched_cache_key, epoched_cache_hash, config, feature_source, mode
        )
        lda_hash = key_hash(lda_key)
        fit_payload = _RUNTIME_LDA_FIT_CACHE.get(lda_hash)
        fit_source: Optional[str] = None
        if fit_payload is None:
            fit_payload = try_load_lda_fit(project_root, lda_key, lda_hash)
            fit_source = "disk" if fit_payload is not None else None
        else:
            fit_source = "memory"
        if fit_payload is not None:
            results = copy.deepcopy(fit_payload)
            msg = (
                "memory"
                if fit_source == "memory"
                else f"disk cache ({lda_hash[:20]}...)"
            )
            logger.info(f"LDA fit (grid+test) loaded from {msg}")
            if permutation_test:
                apply_lda_permutation_to_results(
                    results, X_train, y_train, groups_train, config, logger
                )
        else:
            results = run_lda_fit_and_test_only(
                X_train,
                y_train,
                groups_train,
                X_test,
                y_test,
                config,
                logger,
                feature_source=feature_source,
            )
            to_store = {k: v for k, v in results.items() if k != "permutation_test"}
            _RUNTIME_LDA_FIT_CACHE[lda_hash] = copy.deepcopy(to_store)
            save_lda_fit(project_root, lda_key, lda_hash, to_store)
            logger.info(
                f"Wrote LDA fit disk cache ({lda_hash[:20]}... under results/classification_cache/lda_no_perm)"
            )
            if permutation_test:
                apply_lda_permutation_to_results(
                    results, X_train, y_train, groups_train, config, logger
                )

    if results is None:
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

        history_horizons = config.classification.get("h") or []
        forecast_horizons = config.classification.get("m") or []

        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Flipped classification run timestamp: {run_ts}")

        for hh, fh in _unique_h_m_pairs(history_horizons, forecast_horizons):
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
        history_horizons = config.classification.get("h") or []
        forecast_horizons = config.classification.get("m") or []

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

        for hh, fh in _unique_h_m_pairs(history_horizons, forecast_horizons):
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
            logger.warning(
                f"No test results and no training setup found for variant: {config.run.variant}. "
                f"Skipping test result generation — classification will use train/val only."
            )
            return
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
