from utils.config import get_config
from utils.logger import setup_logger, get_logger
import argparse
from training.components.trainer import Trainer


def train(config):
    logger = get_logger()
    logger.info("Initializing training...")

    fast = getattr(config.model, "fast", False)
    reuse_splits = getattr(config.model, "reuse_splits", False)

    trainer = Trainer(config)
    trainer.split_data(reuse_splits=reuse_splits)
    val_results = trainer.train(fast=fast)

    logger.info(f"Validation results: {val_results}")
    logger.info("Training completed successfully!")


def main(args):
    config = get_config(args.config)
    logger = setup_logger(config.results.log_dir, name=__file__)

    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Config content:\n{config}")
    train(config)

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model using specified configuration."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    args = parser.parse_args()

    main(args)
