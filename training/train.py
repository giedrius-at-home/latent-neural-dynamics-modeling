import argparse
from pathlib import Path

from utils.config import get_config, Config
from utils.logger import setup_logger, get_logger
from training.components.trainer import Trainer


def train(config: Config, model_dbs: str):
    logger = get_logger()
    logger.info("Initializing trainer...")

    fast = config.framework.params.fast
    reuse_splits = config.framework.params.reuse_splits

    trainer = Trainer(config, model_dbs)
    trainer.split_data(reuse_splits=reuse_splits)
    return trainer.train(fast=fast)


def main(args):
    config = get_config(args.config)
    variant_name = f"{config.experiment.name}_dbs_{args.dbs}"
    log_dir = Path(f"results/{config.framework.name}/{variant_name}/logs")
    logger = setup_logger(log_dir, name=__file__)

    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Config content:\n{config}")
    train(config, args.dbs)
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a framework model for a given DBS condition."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the run YAML configuration."
    )
    parser.add_argument(
        "--dbs",
        type=str,
        required=True,
        choices=["both", "on", "off"],
        help="DBS condition to train on.",
    )
    args = parser.parse_args()
    main(args)
