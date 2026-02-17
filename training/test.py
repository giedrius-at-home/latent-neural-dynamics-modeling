from utils.config import get_config
from utils.logger import setup_logger, get_logger
import argparse
from training.components.tester import Tester
from utils.miscellaneous import get_latest_timestamp


def test(config, run_timestamp=None):
    logger = get_logger()
    logger.info("Initializing tester...")

    if run_timestamp is None:
        logger.info("No run timestamp provided, using the latest available run.")
        run_timestamp = get_latest_timestamp(config.results.save_dir)

    logger.info(f"Using run timestamp: {run_timestamp}")
    tester = Tester(config, run_timestamp=run_timestamp)
    tester.run_predictions()

    for split, res in tester.results.items():
        # Log prediction correlations for Y
        means = res.get("pearson_mean", [])
        if len(means) > 0:
            valid = [m for m in means if m == m]  # filter NaN
            avg = sum(valid) / len(valid) if valid else float("nan")
        else:
            avg = float("nan")
        logger.info(f"Split={split}: avg Pearson (Y predictions) over trials={avg:.4f}")

        # Log prediction correlations for Z
        means_z = res.get("pearson_mean_Z", [])
        if means_z is not None and len(means_z) > 0:
            valid_z = [m for m in means_z if m == m]  # filter NaN
            avg_z = sum(valid_z) / len(valid_z) if valid_z else float("nan")
            if avg_z == avg_z:  # check not NaN
                logger.info(
                    f"Split={split}: avg Pearson (Z predictions) over trials={avg_z:.4f}"
                )

        # Log forecast correlations for Y
        forecast_y_mean = res.get("pearson_overall_mean", float("nan"))
        if forecast_y_mean == forecast_y_mean:  # check not NaN
            logger.info(f"Split={split}: Pearson (Y forecast)={forecast_y_mean:.4f}")

        # Log forecast correlations for Z
        forecast_z_mean = res.get("pearson_overall_mean_Z", float("nan"))
        if forecast_z_mean == forecast_z_mean:  # check not NaN
            logger.info(f"Split={split}: Pearson (Z forecast)={forecast_z_mean:.4f}")

    logger.info("Testing completed successfully!")

    tester.save_results()
    logger.info("Saved results.")
    tester.compute_and_save_stats()
    logger.info("Saved statistics.")


def main(args):
    config = get_config(args.config)
    logger = setup_logger(config.results.log_dir, name=__file__)

    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Config content:\n{config}")
    test(config, run_timestamp=args.run)

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run predictions using a saved model and output quick metrics."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    parser.add_argument(
        "--run",
        type=str,
        required=False,
        help="Run timestamp to load (e.g., 20251103_104200 or val_results_20251103_104200). If omitted, latest is used.",
    )
    args = parser.parse_args()

    main(args)
