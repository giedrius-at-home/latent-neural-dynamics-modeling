import polars as pl
from pathlib import Path

from utils.file_handling import list_files, load_mat_into_dict
from utils.config import Config

from utils.ieeg import (
    preprocess_ieeg,
    filter_recording,
    process_and_resample,
    process_dual_band,
)

from scipy.io import savemat

import numpy as np
from utils.logger import get_logger


def read_tsv(path: Path, unique: bool = True) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        null_values="n/a",
    )
    if unique:
        df = df.unique(maintain_order=True)
    return df


def read_tsv_to_struct(path: Path) -> pl.Series:
    df = read_tsv(path, unique=False)

    return df.to_struct()


def read_tsv_to_dict(path: Path) -> dict:
    df = read_tsv(path, unique=False)

    return df.to_dict(as_series=True)


def add_modality_path(participants: pl.DataFrame, modality: str) -> pl.DataFrame:

    participants_ = participants.with_columns(
        pl.concat_str(
            [
                pl.col("session_path"),
                pl.lit(modality),
            ],
            separator="/",
        ).alias(f"{modality}_path"),
    )

    return participants_


def explode_files(
    participants: pl.DataFrame, folder: str, out_file_col: str
) -> pl.DataFrame:

    participants_ = participants.with_columns(
        pl.col(folder)
        .map_elements(
            lambda folder: list_files(Path(folder)), return_dtype=pl.List(pl.String)
        )
        .alias(out_file_col)
    ).explode(pl.col(out_file_col))

    return participants_


def split_file_path(
    participants: pl.DataFrame,
    modality: str,
    positions: list[tuple[str, int, pl.DataType]],
) -> pl.DataFrame:

    from utils.logger import get_logger

    logger = get_logger()

    participants_ = participants.with_columns(
        pl.col(f"{modality}_file")
        .str.split(by="/")
        .list.get(-1)
        .str.split("_")
        .alias("split_file")
    )

    participants_ = participants_.with_columns(
        pl.col("split_file").list.get(-1).str.split(".").list.get(0).alias("type"),
        pl.col("split_file")
        .list.get(-1)
        .str.split(".")
        .list.get(-1)
        .alias("data_format"),
    )

    for part, indx, dtype in positions:
        logger.info(f"Extracting '{part}' from filename at index {indx} as {dtype}")
        participants_ = participants_.with_columns(
            pl.col("split_file")
            .list.get(indx)
            .str.split("-")
            .list.get(-1)
            .cast(dtype, strict=False)
            .alias(part),
        )

    return participants_.drop("split_file")


def __create_and_polars_filter(**kwargs) -> bool:
    first_key, first_value = next(iter(kwargs.items()))
    filter_ = pl.col(first_key) == first_value

    for key, value in list(kwargs.items())[1:]:
        filter_ = filter_ & (pl.col(key) == value)

    return filter_


def remove_rows_with(table: pl.DataFrame, **kwargs) -> pl.DataFrame:
    filter_ = __create_and_polars_filter(**kwargs)
    table_ = table.filter(~filter_)

    return table_


def keep_rows_with(table: pl.DataFrame, **kwargs) -> pl.DataFrame:
    filter_ = __create_and_polars_filter(**kwargs)
    table_ = table.filter(filter_)

    return table_


def get_trial(
    participants: pl.DataFrame,
    participant_id: str,
    session: int,
    block: int,
    trial: int,
    columns: list[str] = None,
    explode: list[str] = None,
) -> pl.DataFrame:

    trial_df = keep_rows_with(
        participants,
        participant_id=participant_id,
        session=session,
        block=block,
        trial=trial,
    )
    if columns:
        trial_df = trial_df.select(columns)
    if explode:
        trial_df = trial_df.explode(explode)
    return trial_df


def load_parquet_as_dict(parquet_file: str) -> dict:

    df_ = pl.read_parquet(parquet_file)

    return df_.to_dict()


def dict_to_struct(data: dict) -> pl.Series:
    return pl.DataFrame(data).select(pl.all().implode()).to_struct()


def remove_chunk_margin(
    participants: pl.DataFrame, col: str, return_as: str
) -> pl.DataFrame:
    participants_ = participants.with_columns(
        (pl.col("chunk_margin") * 1000).alias("chunk_margin_ts")
    )

    participants_ = participants_.with_columns(
        pl.col(col)
        .list.slice(
            pl.col("chunk_margin_ts"),
            pl.col("trial_length_ts") - 2 * pl.col("chunk_margin_ts"),
        )
        .alias(return_as)
    ).drop("chunk_margin_ts")

    return participants_


def read_and_implode_parquet(path: str) -> pl.Series:
    if not Path(path).exists():
        return None
    df = pl.read_parquet(path)
    return df.select(pl.all().implode()).to_struct()


def band_pass_resample(
    participants: pl.DataFrame, config: Config, iEEG_SCHEMA: pl.Struct
):
    """CODE IF PARTIAL PREPROCESSING IS NOT DONE
    resampled_freq = config.ieeg_process.resampled_freq
    low_freq = config.ieeg_process.low_freq
    high_freq = config.ieeg_process.high_freq
    notch_freqs = config.ieeg_process.notch_freqs

    ieeg_headers_files = participants["ieeg_headers_file"].to_list()

    participants.write_parquet(
        save_dir / "participants_intermediate.parquet",
        partition_by=["participant_id", "session", "run"],
    )
    del participants

    for iegg_hf in ieeg_headers_files:
        base_file = iegg_hf.split("/")[-1].split(".")[0]
        ieeg_dict = preprocess_ieeg(
            iegg_hf, resampled_freq, low_freq, high_freq, notch_freqs
        )

        print(save_dir / f"{base_file}.mat")

        savemat(save_dir / f"{base_file}.mat", ieeg_dict)

    participants_ = pl.read_parquet(save_dir / "participants_intermediate.parquet")

    participants_ = participants_.with_columns(
        pl.col("ieeg_headers_file")
        .str.split("/")
        .list.get(-1)
        .str.split(".")
        .list.get(0)
        .alias("iegg_hf_base_file")
    )
    participants_ = participants_.with_columns(
        pl.concat_str(
            pl.lit(str(save_dir)),
            pl.concat_str(pl.col("iegg_hf_base_file"), pl.lit("mat"), separator="."),
            separator="/",
        ).alias("iegg_mat_file")
    ).drop("iegg_hf_base_file")

    participants_ = participants_.with_columns(
        pl.col("iegg_mat_file")
        .map_elements(load_mat_into_dict, return_dtype=pl.Object)
        .alias("ieeg_raw")
    )

    for ieeg_field in iEEG_SCHEMA.fields:
        participants_ = participants_.with_columns(
            pl.col("ieeg_raw")
            .map_elements(lambda ieeg_raw: ieeg_raw[ieeg_field.name][0], ieeg_field.dtype)
            .alias(ieeg_field.name)
        )

    return participants_.drop("ieeg_raw")
    """

    participants_ = (
        participants.with_columns(
            pl.col("ieeg_parquet")
            .map_elements(read_and_implode_parquet, return_dtype=pl.List(iEEG_SCHEMA))
            .alias("ieeg_data")
        )
        .explode("ieeg_data")
        .unnest("ieeg_data")
        .drop("sfreq")
    )

    original_sfreq = 1000
    target_sfreq = config.ieeg_process.resampled_freq
    envelope_cutoff = config.ieeg_process.envelope_cutoff

    for ieeg_field in iEEG_SCHEMA.fields:
        if ieeg_field.name == "sfreq":
            continue

        def _process_dual(r, field_name=ieeg_field.name):
            if r is None or len(r) == 0:
                return {"raw": [], "envelope": []}
            scale_factor = float(getattr(config.ieeg_process, 'scale_factor', 1.0))
            raw, envelope = process_dual_band(
                r,
                low_freq=config.ieeg_process.low_freq,
                envelope_cutoff=envelope_cutoff,
                high_freq=config.ieeg_process.high_freq,
                notch_freqs=config.ieeg_process.notch_freqs,
                original_sfreq=original_sfreq,
                target_sfreq=target_sfreq,
                scale_factor=scale_factor,
            )
            return {"raw": raw, "envelope": envelope}

        participants_ = participants_.with_columns(
            pl.col(ieeg_field.name)
            .map_elements(_process_dual, return_dtype=pl.Object)
            .alias(f"_dual_{ieeg_field.name}")
        )

        participants_ = participants_.with_columns(
            pl.col(f"_dual_{ieeg_field.name}")
            .map_elements(lambda x: x["raw"], return_dtype=pl.List(pl.Float32))
            .alias(ieeg_field.name),
            pl.col(f"_dual_{ieeg_field.name}")
            .map_elements(lambda x: x["envelope"], return_dtype=pl.List(pl.Float32))
            .alias(f"{ieeg_field.name}_envelope"),
        ).drop(f"_dual_{ieeg_field.name}")

    return participants_


def stack_columns(participants: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    segments_per_col = []
    for col in cols:
        lists = participants[col].to_list()
        arrs = [np.asarray(x, dtype=np.float64) for x in lists]
        segments_per_col.append(arrs)

    concatenated = [np.concatenate(col_segs) for col_segs in segments_per_col]

    matrix = np.vstack(concatenated)

    return matrix, segments_per_col


def get_scalar_value(df: pl.DataFrame, col_name: str):
    if col_name not in df.columns:
        return None
    val = df[col_name][0]
    if isinstance(val, pl.Series):
        return val.item()
    return val


def convert_series_to_list(series_list: list):
    return [s.to_list() if isinstance(s, pl.Series) else s for s in series_list]


def set_polars_config():
    print("Loading Polars configuration from utils/polars.py")
    pl.Config.set_tbl_width_chars(-1)
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_fmt_str_lengths(2000)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
