from utils.polars import set_polars_config
from utils.logger import logger_executor
from utils.config import get_config, Config


def initialize_preprocessing(config_path: str) -> Config:
    config_ = get_config(config_path)
    logger_executor(config_.logger_directory, name=config_.name)

    set_polars_config()

    return config_
