import polars as pl
from utils.polars import (
    read_tsv_to_dict,
    keep_rows_with,
    add_modality_path,
    explode_files,
    split_file_path,
    keep_rows_with,
)
from utils.config import Config

from utils.file_handling import read_json
from pathlib import Path
from utils.logger import get_logger


def _get_motion_coordinates(motion: pl.DataFrame) -> pl.DataFrame:
    motion_coords = keep_rows_with(motion, type="motion", data_format="tsv")
    motion_coords = motion_coords.with_columns(
        pl.col("motion_file")
        .map_elements(read_tsv_to_dict, return_dtype=pl.Object)
        .alias("motion_coordinates")
    )
    motion_coords = motion_coords.with_columns(
        pl.col("motion_coordinates")
        .map_elements(lambda md: md["x"], pl.List(pl.Int64))
        .alias("x"),
        pl.col("motion_coordinates")
        .map_elements(lambda md: md["y"], pl.List(pl.Int64))
        .alias("y"),
    ).drop("motion_coordinates")

    motion_coords = motion_coords.sort(by=["participant_id", "session", "run", "chunk"])
    logger = get_logger()
    logger.info(
        f"Motion coordinates table constructed with {motion_coords.height} rows."
    )
    return motion_coords


def _get_motion_dbs_cond(motion: pl.DataFrame) -> pl.DataFrame:
    motion_dbs_cond = (
        keep_rows_with(motion, type="motion", data_format="json")
        .filter(pl.col("chunk").is_null())
        .drop("chunk", "type", "data_format")
    )

    motion_dbs_cond = motion_dbs_cond.with_columns(
        pl.col("motion_file")
        .map_elements(lambda mf: read_json(mf)["dbs_stim"], return_dtype=pl.String)
        .alias("dbs_stim")
    ).drop("motion_file")

    return motion_dbs_cond


def construct_motion_table(participants: pl.DataFrame, config: Config) -> pl.DataFrame:
    motion_ = participants.unique(["participant_id", "session_path"])

    #### DUE TO PARTIAL PREPROCESSED DATA
    motion_ = motion_.select("participant_id", "session_path")
    motion_ = motion_.with_columns(
        pl.col("session_path")
        .str.split(by="/")
        .list.tail(2)
        .list.join(separator="/")
        .map_elements(
            lambda s: str(Path(config.data_directory).joinpath(s)),
            return_dtype=pl.String,
        )
        .alias("session_path")
    )
    ####

    motion_ = add_modality_path(motion_, "motion")
    motion_ = explode_files(motion_, "motion_path", "motion_file")
    motion_ = split_file_path(
        motion_,
        "motion",
        [
            ("session", 1, pl.UInt64),
            ("task", 2, pl.String),
            ("run", 3, pl.UInt64),
            ("chunk", 4, pl.UInt64),
        ],
    )
    motion_ = keep_rows_with(motion_, task="copydraw").drop("task")

    motion_ = _get_motion_coordinates(motion_)
    motion_ = motion_.rename({"run": "block", "chunk": "trial"})

    return motion_
