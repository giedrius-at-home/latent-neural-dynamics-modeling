import argparse
from pathlib import Path

from utils.config import get_config, Config
from utils.logger import setup_logger, get_logger
from utils.miscellaneous import get_latest_timestamp
from training.components.tester import Tester


def test(
    config: Config,
    model_dbs: str,
    run_timestamp=None,
    incremental=False,
    splits=None,
    h=None,
    m=None,
):
    logger = get_logger()
    logger.info("Initializing tester...")

    if run_timestamp is None:
        logger.info("No run timestamp provided, using the latest available run.")
        variant_name = f"{config.experiment.name}_dbs_{model_dbs}"
        model_dir = Path(f"results/{config.framework.name}/{variant_name}")
        run_timestamp = get_latest_timestamp(str(model_dir))

    logger.info(f"Using run timestamp: {run_timestamp}")
    tester = Tester(config, model_dbs, run_ts=run_timestamp, h=h, m=m)

    if incremental:
        logger.info("Running incremental (trial-by-trial) predictions...")
        tester.run_predictions_incremental(splits=splits)
    else:
        if splits:
            logger.info(f"Running selective predictions on splits: {splits}")
            tester.run_predictions_selective(splits)
        else:
            tester.run_predictions()

        for split, res in tester.results.items():
            means = res.get("pearson_mean", [])
            if len(means) > 0:
                valid = [mv for mv in means if mv == mv]
                avg = sum(valid) / len(valid) if valid else float("nan")
            else:
                avg = float("nan")
            logger.info(
                f"Split={split}: avg Pearson (Y predictions) over trials={avg:.4f}"
            )

            means_z = res.get("pearson_mean_Z", [])
            if means_z is not None and len(means_z) > 0:
                valid_z = [mv for mv in means_z if mv == mv]
                avg_z = sum(valid_z) / len(valid_z) if valid_z else float("nan")
                if avg_z == avg_z:
                    logger.info(
                        f"Split={split}: avg Pearson (Z predictions) over trials={avg_z:.4f}"
                    )

            forecast_y_mean = res.get("pearson_overall_mean", float("nan"))
            if forecast_y_mean == forecast_y_mean:
                logger.info(
                    f"Split={split}: Pearson (Y forecast)={forecast_y_mean:.4f}"
                )

            forecast_z_mean = res.get("pearson_overall_mean_Z", float("nan"))
            if forecast_z_mean == forecast_z_mean:
                logger.info(
                    f"Split={split}: Pearson (Z forecast)={forecast_z_mean:.4f}"
                )

        tester.save_results()
        logger.info("Saved results.")

    logger.info("Testing completed successfully!")
    tester.compute_and_save_stats()
    logger.info("Saved statistics.")


def main(args):
    config = get_config(args.config)
    variant_name = f"{config.experiment.name}_dbs_{args.model_dbs}"
    log_suffix = f"forecast_h{args.h:g}" if args.h is not None else "inference"
    log_dir = Path(f"results/{config.framework.name}/{variant_name}/{log_suffix}/logs")
    logger = setup_logger(log_dir, name=__file__)

    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Config content:\n{config}")
    test(
        config,
        model_dbs=args.model_dbs,
        run_timestamp=args.run,
        incremental=args.incremental,
        splits=args.splits,
        h=args.h,
        m=args.m,
    )
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run predictions using a saved model and output quick metrics."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the run YAML configuration."
    )
    parser.add_argument(
        "--model-dbs",
        type=str,
        required=True,
        choices=["both", "on", "off"],
        help="DBS condition the model was trained on.",
    )
    parser.add_argument(
        "--run",
        type=str,
        required=False,
        help="Run timestamp to load. If omitted, latest is used.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Process and save one trial at a time (resumable on crash).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=None,
        help="Subset of splits to run predictions on. If omitted, runs all three.",
    )
    parser.add_argument(
        "--forecast-h",
        dest="h",
        type=float,
        default=None,
        help="Forecast history length in seconds. If set, runs forecast mode.",
    )
    parser.add_argument(
        "--forecast-m",
        dest="m",
        type=float,
        default=None,
        help="Forecast horizon in seconds.",
    )
    args = parser.parse_args()
    main(args)
