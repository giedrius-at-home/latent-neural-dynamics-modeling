import polars as pl
import numpy as np
from pathlib import Path

from utils.ieeg import (
    epoch_trials,
    calculate_psd_welch,
)
from utils.polars import (
    # read_tsv,
    # add_modality_path,
    # split_file_path,
    # remove_rows_with,
    # explode_files,
    # keep_rows_with,
    band_pass_resample,
    # stack_columns,
)

# from .events import construct_events_table
from .motion import construct_motion_table
from utils.file_handling import get_child_subchilds_tuples

from utils.config import Config
from utils.logger import get_logger
from utils.motion import (
    compute_tracing_speeds,
    compute_tracing_acceleration,
    compute_tracing_jerk,
    interpolate,
    _smooth_signal,
)

LFP_SCHEMA = pl.Struct(
    [
        pl.Field("LFP_1", pl.List(pl.Float32)),
        pl.Field("LFP_2", pl.List(pl.Float32)),
        pl.Field("LFP_3", pl.List(pl.Float32)),
        pl.Field("LFP_4", pl.List(pl.Float32)),
        pl.Field("LFP_5", pl.List(pl.Float32)),
        pl.Field("LFP_6", pl.List(pl.Float32)),
        pl.Field("LFP_7", pl.List(pl.Float32)),
        pl.Field("LFP_8", pl.List(pl.Float32)),
        pl.Field("LFP_9", pl.List(pl.Float32)),
        pl.Field("LFP_10", pl.List(pl.Float32)),
        pl.Field("LFP_11", pl.List(pl.Float32)),
        pl.Field("LFP_12", pl.List(pl.Float32)),
        pl.Field("LFP_13", pl.List(pl.Float32)),
        pl.Field("LFP_14", pl.List(pl.Float32)),
        pl.Field("LFP_15", pl.List(pl.Float32)),
        pl.Field("LFP_16", pl.List(pl.Float32)),
    ]
)

ECOG_SCHEMA = pl.Struct(
    [
        pl.Field("ECOG_1", pl.List(pl.Float32)),
        pl.Field("ECOG_2", pl.List(pl.Float32)),
        pl.Field("ECOG_3", pl.List(pl.Float32)),
        pl.Field("ECOG_4", pl.List(pl.Float32)),
    ]
)

EOG_SCHEMA = pl.Struct(
    [
        pl.Field("EOG_1", pl.List(pl.Float32)),
        pl.Field("EOG_2", pl.List(pl.Float32)),
        pl.Field("EOG_3", pl.List(pl.Float32)),
        pl.Field("EOG_4", pl.List(pl.Float32)),
    ]
)

iEEG_SCHEMA = pl.Struct(
    [
        *LFP_SCHEMA.fields,
        *ECOG_SCHEMA.fields,
        *EOG_SCHEMA.fields,
        pl.Field("sfreq", pl.List(pl.Float32)),
    ]
)


def _add_full_data(participants: pl.DataFrame, config: Config) -> pl.DataFrame:

    logger = get_logger()

    """CODE IF PARTIAL PREPROCESSING IS NOT DONE
    participants = read_tsv(data_path / config.participants_table_name)
    participants = participants.filter(
        pl.col("participant_id").str.starts_with("sub-PD")
    )
    participants = participants.drop("age", "sex", "hand", "weight", "height")

    participants = participants.with_columns(
        pl.concat_str(
            pl.lit(str(data_path)), pl.col("participant_id"), separator="/"
        ).alias("participant_path")
    )
    participants = explode_files(participants, "participant_path", "session_path")
    """
    logger.info("Adding full data")
    ieeg_participants = _add_ieeg_data(participants, config)
    logger.info(
        participants.group_by(["participant_id", "session", "block", "trials"]).len()
    )
    logger.info("Loaded iEEG data")
    lfp_channels = [field.name for field in LFP_SCHEMA.fields]
    ecog_channels = [field.name for field in ECOG_SCHEMA.fields]
    all_channels = lfp_channels + ecog_channels

    ieeg_participants = apply_car(ieeg_participants, lfp_channels, config)
    ieeg_participants = apply_car(ieeg_participants, ecog_channels, config)

    motion_participants = construct_motion_table(ieeg_participants, config)
    logger.info("Loaded motion data")

    ieeg_participants = (
        (
            ieeg_participants.explode("trials")
            .with_columns(
                pl.col("onsets")
                .list.get(pl.col("trials") - 1, null_on_oob=True)
                .alias("onset")
            )
            .drop("onsets", "yscores", "trial_index", strict=False)
        )
        .with_columns(pl.col("trials").alias("trial"))
        .drop("trials")
    )
    participants = ieeg_participants.join(
        motion_participants,
        on=["participant_id", "session", "block", "trial"],
        how="left",
    )
    logger.info(
        participants.group_by(["participant_id", "session", "block", "trial"]).len()
    )
    participants = participants.drop(
        "participant_path",
        "session_path",
        "participant_path_right",
        "session_path_right",
        "ieeg_path",
        "ieeg_file",
        "ieeg_headers_file",
        "motion_path",
        "motion_file",
        "type",
        "data_format",
        strict=False,
    )

    participants = _chunk_recordings(
        participants,
        config.ieeg_process.chunk_margin,
        config.ieeg_process.resampled_freq,
    )

    for channel in all_channels:
        participants = participants.with_columns(
            pl.col(channel)
            .map_elements(
                epoch_trials,
                return_dtype=pl.List(pl.List(pl.Float64)),
            )
            .alias(f"{channel}_epochs")
        )

    for channel in all_channels:
        participants = (
            participants.with_columns(
                pl.col(f"{channel}_epochs")
                .map_elements(
                    lambda x: calculate_psd_welch(
                        x,
                        sfreq=config.ieeg_process.resampled_freq,
                        low_freq=config.ieeg_process.low_freq,
                        high_freq=config.ieeg_process.high_freq,
                    ),
                    return_dtype=pl.Object,
                )
                .alias(f"{channel}_psd")
            )
            .with_columns(
                pl.col(f"{channel}_psd")
                .map_elements(lambda x: x[0], return_dtype=pl.List(pl.Float64))
                .alias(f"{channel}_psd_freq"),
                pl.col(f"{channel}_psd")
                .map_elements(lambda x: x[1], return_dtype=pl.List(pl.List(pl.Float64)))
                .alias(f"{channel}_psd_values"),
            )
            .drop(f"{channel}_psd")
        )

    return participants


def construct_participants_table(config: Config):
    logger = get_logger()

    data_path = Path(config.data_directory)
    save_path = Path(config.save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    participants_partitions = get_child_subchilds_tuples(
        data_path / config.participants_intermediate_table_name
    )

    for p_part in participants_partitions:
        root, participant_id, session, block = p_part
        p_partition_path = data_path / root / participant_id / session / block / "*"
        participants = pl.read_parquet(p_partition_path)

        logger.info(f"Loaded participants from: {p_partition_path}")

        participants = _add_full_data(participants, config)

        participants = participants.select(
            "participant_id",
            "session",
            "block",
            "trial",
            "onset",
            "margined_onset",
            "margined_duration",
            "time",
            "time_original",
            "motion_time",
            "original_length_ts",
            "start_ts",
            pl.col("chunk_length_ts").alias("trial_length_ts"),
            "chunk_margin",
            "stim",
            pl.col("^LFP_.*$"),
            pl.col("^ECOG_.*$"),
            "x",
            "y",
            "tracing_speed",
            "tracing_speed_x",
            "tracing_speed_y",
            "tracing_speed_magnitude",
            "tracing_acceleration",
            "tracing_acceleration_x",
            "tracing_acceleration_y",
            "tracing_acceleration_magnitude",
            "tracing_jerk",
            "tracing_jerk_x",
            "tracing_jerk_y",
            "tracing_jerk_magnitude",
        )
        participants.write_parquet(
            save_path / config.output_participants_table_name,
            partition_by=["participant_id", "session", "block"],
        )
        logger.info(f"Saved to {save_path / config.output_participants_table_name}")


def _add_ieeg_data(participants: pl.DataFrame, config: Config) -> pl.DataFrame:
    """CODE IF PARTIAL PREPROCESSING IS NOT DONE

    participants = add_modality_path(participants, "ieeg")

    participants = explode_files(participants, "ieeg_path", "ieeg_file")

    participants = split_file_path(
        participants,
        "ieeg",
        [("session", 1, pl.UInt64), ("task", 2, pl.String), ("run", 3, pl.UInt64)],
    )
    participants = keep_rows_with(participants, task="copydraw").drop("task")
    participants = remove_rows_with(participants, type="channels", data_format="tsv")

    events = construct_events_table(participants)

    participants = participants.join(
        events, on=["participant_id", "session", "run"], how="left"
    )

    participants = remove_rows_with(participants, type="events", data_format="tsv")
    headers = keep_rows_with(participants, type="ieeg", data_format="vhdr").select(
        "participant_id",
        "session",
        "run",
        pl.col("ieeg_file").alias("ieeg_headers_file"),
    )

    participants = participants.join(
        headers, on=["participant_id", "session", "run"], how="left"
    )

    participants = remove_rows_with(participants, type="ieeg", data_format="vhdr")
    participants = remove_rows_with(participants, type="ieeg", data_format="vmkr")

    participants = participants.drop(
        "type", "data_format", "channels_info_right", strict=False
    )
    """
    participants = band_pass_resample(participants, config, iEEG_SCHEMA)

    return participants


def _chunk_recordings(
    participants: pl.DataFrame, chunk_margin: int, sfreq: int
) -> pl.DataFrame:

    participants_ = participants.with_columns(
        (pl.col("onset") - chunk_margin).alias("margined_onset"),
        (pl.col("trial_time") + 2 * chunk_margin).alias("margined_duration"),
    )

    participants_ = participants_.with_columns(
        (pl.col("margined_onset") * sfreq).cast(pl.UInt32).alias("start_ts"),
        (pl.col("margined_duration") * sfreq).cast(pl.UInt32).alias("chunk_length_ts"),
        (pl.col("trial_time") * sfreq).cast(pl.UInt32).alias("original_length_ts"),
    ).with_columns(
        pl.int_ranges(0, pl.col("chunk_length_ts"), dtype=pl.UInt32)
        .truediv(sfreq)
        .add(pl.col("margined_onset"))
        .alias("time"),
        pl.int_ranges(0, pl.col("original_length_ts"), dtype=pl.UInt32)
        .truediv(sfreq)
        .add(pl.col("onset"))
        .alias("time_original"),
    )

    for ieeg_field in iEEG_SCHEMA.fields:
        if ieeg_field.name == "sfreq":
            continue
        participants_ = participants_.with_columns(
            pl.col(ieeg_field.name).list.slice(
                pl.col("start_ts"), pl.col("chunk_length_ts")
            )
        )

    participants_ = participants_.with_columns(
        pl.lit(chunk_margin).alias("chunk_margin")
    )

    participants_ = participants_.with_columns(
        pl.when(pl.col("x").list.len() > 0)
        .then(
            (
                pl.int_ranges(0, pl.col("x").list.len())
                * (pl.col("trial_time") / pl.col("x").list.len())
            )
            + pl.col("onset")
        )
        .alias("motion_time")
    )

    from utils.logger import get_logger

    logger = get_logger()
    logger.info(
        f"""There are null values in motion_time column: 
        {participants_.select("participant_id", "session", "block", "trial", "motion_time").filter(pl.col('motion_time').is_null())}"""
    )

    logger.info(
        f"""There are null values within motion_time column: 
        {participants_.select("participant_id", "session", "block", "trial", "motion_time").filter(pl.col('motion_time').list.contains(None))}"""
    )

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("motion_time").is_not_null()
            & (pl.col("motion_time").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(pl.col("x"), pl.col("y"), "motion_time").map_elements(
                lambda s: compute_tracing_speeds(s["x"], s["y"], s["motion_time"]),
                return_dtype=pl.Object,
            )
        )
        .alias("_speed_results")
    )

    participants_ = participants_.with_columns(
        pl.col("_speed_results")
        .map_elements(
            lambda r: r["tracing_speed"] if r is not None else None,
            return_dtype=pl.List(pl.Float64),
        )
        .alias("tracing_speed"),
        pl.col("_speed_results")
        .map_elements(
            lambda r: r["tracing_speed_x"] if r is not None else None,
            return_dtype=pl.List(pl.Float64),
        )
        .alias("tracing_speed_x"),
        pl.col("_speed_results")
        .map_elements(
            lambda r: r["tracing_speed_y"] if r is not None else None,
            return_dtype=pl.List(pl.Float64),
        )
        .alias("tracing_speed_y"),
        pl.col("_speed_results")
        .map_elements(
            lambda r: r["tracing_speed_magnitude"] if r is not None else None,
            return_dtype=pl.List(pl.Float64),
        )
        .alias("tracing_speed_magnitude"),
    ).drop("_speed_results")

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_speed_x").is_not_null()
            & (pl.col("tracing_speed_x").list.drop_nulls().list.len() > 0)
            & pl.col("tracing_speed_y").is_not_null()
            & (pl.col("tracing_speed_y").list.drop_nulls().list.len() > 0)
            & pl.col("motion_time").is_not_null()
            & (pl.col("motion_time").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_speed_x"), pl.col("tracing_speed_y"), "motion_time"
            ).map_elements(
                lambda s: compute_tracing_acceleration(
                    s["tracing_speed_x"], s["tracing_speed_y"], s["motion_time"]
                ),
                return_dtype=pl.Object,
            )
        )
        .alias("_accel_results")
    )

    participants_ = participants_.with_columns(
        pl.col("_accel_results")
        .map_elements(
            lambda r: r["tracing_acceleration"] if r is not None else None,
            return_dtype=pl.List(pl.Float64),
        )
        .alias("tracing_acceleration"),
        pl.col("_accel_results")
        .map_elements(
            lambda r: r["tracing_acceleration_x"] if r is not None else None,
            return_dtype=pl.List(pl.Float64),
        )
        .alias("tracing_acceleration_x"),
        pl.col("_accel_results")
        .map_elements(
            lambda r: r["tracing_acceleration_y"] if r is not None else None,
            return_dtype=pl.List(pl.Float64),
        )
        .alias("tracing_acceleration_y"),
        pl.col("_accel_results")
        .map_elements(
            lambda r: r["tracing_acceleration_magnitude"] if r is not None else None,
            return_dtype=pl.List(pl.Float64),
        )
        .alias("tracing_acceleration_magnitude"),
    ).drop("_accel_results")

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_acceleration_x").is_not_null()
            & (pl.col("tracing_acceleration_x").list.drop_nulls().list.len() > 0)
            & pl.col("tracing_acceleration_y").is_not_null()
            & (pl.col("tracing_acceleration_y").list.drop_nulls().list.len() > 0)
            & pl.col("motion_time").is_not_null()
            & (pl.col("motion_time").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_acceleration_x"),
                pl.col("tracing_acceleration_y"),
                "motion_time",
            ).map_elements(
                lambda s: compute_tracing_jerk(
                    s["tracing_acceleration_x"],
                    s["tracing_acceleration_y"],
                    s["motion_time"],
                ),
                return_dtype=pl.Object,
            )
        )
        .alias("_jerk_results")
    )

    participants_ = participants_.with_columns(
        pl.col("_jerk_results")
        .map_elements(
            lambda r: r["tracing_jerk"] if r is not None else None,
            return_dtype=pl.List(pl.Float64),
        )
        .alias("tracing_jerk"),
        pl.col("_jerk_results")
        .map_elements(
            lambda r: r["tracing_jerk_x"] if r is not None else None,
            return_dtype=pl.List(pl.Float64),
        )
        .alias("tracing_jerk_x"),
        pl.col("_jerk_results")
        .map_elements(
            lambda r: r["tracing_jerk_y"] if r is not None else None,
            return_dtype=pl.List(pl.Float64),
        )
        .alias("tracing_jerk_y"),
        pl.col("_jerk_results")
        .map_elements(
            lambda r: r["tracing_jerk_magnitude"] if r is not None else None,
            return_dtype=pl.List(pl.Float64),
        )
        .alias("tracing_jerk_magnitude"),
    ).drop("_jerk_results")

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_speed").is_not_null()
            & (pl.col("tracing_speed").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_speed"), pl.col("original_length_ts")
            ).map_elements(
                lambda s: interpolate(s["tracing_speed"], s["original_length_ts"]),
                return_dtype=pl.List(pl.Float64),
            )
        )
        .alias("tracing_speed")
    )

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_speed_x").is_not_null()
            & (pl.col("tracing_speed_x").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_speed_x"), pl.col("original_length_ts")
            ).map_elements(
                lambda s: interpolate(s["tracing_speed_x"], s["original_length_ts"]),
                return_dtype=pl.List(pl.Float64),
            )
        )
        .alias("tracing_speed_x")
    )

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_speed_y").is_not_null()
            & (pl.col("tracing_speed_y").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_speed_y"), pl.col("original_length_ts")
            ).map_elements(
                lambda s: interpolate(s["tracing_speed_y"], s["original_length_ts"]),
                return_dtype=pl.List(pl.Float64),
            )
        )
        .alias("tracing_speed_y")
    )

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_speed_magnitude").is_not_null()
            & (pl.col("tracing_speed_magnitude").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_speed_magnitude"), pl.col("original_length_ts")
            ).map_elements(
                lambda s: interpolate(
                    s["tracing_speed_magnitude"], s["original_length_ts"]
                ),
                return_dtype=pl.List(pl.Float64),
            )
        )
        .alias("tracing_speed_magnitude")
    )

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_acceleration").is_not_null()
            & (pl.col("tracing_acceleration").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_acceleration"), pl.col("original_length_ts")
            ).map_elements(
                lambda s: interpolate(
                    s["tracing_acceleration"], s["original_length_ts"]
                ),
                return_dtype=pl.List(pl.Float64),
            )
        )
        .alias("tracing_acceleration")
    )

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_acceleration_x").is_not_null()
            & (pl.col("tracing_acceleration_x").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_acceleration_x"), pl.col("original_length_ts")
            ).map_elements(
                lambda s: interpolate(
                    s["tracing_acceleration_x"], s["original_length_ts"]
                ),
                return_dtype=pl.List(pl.Float64),
            )
        )
        .alias("tracing_acceleration_x")
    )

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_acceleration_y").is_not_null()
            & (pl.col("tracing_acceleration_y").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_acceleration_y"), pl.col("original_length_ts")
            ).map_elements(
                lambda s: interpolate(
                    s["tracing_acceleration_y"], s["original_length_ts"]
                ),
                return_dtype=pl.List(pl.Float64),
            )
        )
        .alias("tracing_acceleration_y")
    )

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_acceleration_magnitude").is_not_null()
            & (
                pl.col("tracing_acceleration_magnitude").list.drop_nulls().list.len()
                > 0
            )
        )
        .then(
            pl.struct(
                pl.col("tracing_acceleration_magnitude"), pl.col("original_length_ts")
            ).map_elements(
                lambda s: interpolate(
                    s["tracing_acceleration_magnitude"], s["original_length_ts"]
                ),
                return_dtype=pl.List(pl.Float64),
            )
        )
        .alias("tracing_acceleration_magnitude")
    )

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_jerk").is_not_null()
            & (pl.col("tracing_jerk").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_jerk"), pl.col("original_length_ts")
            ).map_elements(
                lambda s: interpolate(s["tracing_jerk"], s["original_length_ts"]),
                return_dtype=pl.List(pl.Float64),
            )
        )
        .alias("tracing_jerk")
    )

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_jerk_x").is_not_null()
            & (pl.col("tracing_jerk_x").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_jerk_x"), pl.col("original_length_ts")
            ).map_elements(
                lambda s: interpolate(s["tracing_jerk_x"], s["original_length_ts"]),
                return_dtype=pl.List(pl.Float64),
            )
        )
        .alias("tracing_jerk_x")
    )

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_jerk_y").is_not_null()
            & (pl.col("tracing_jerk_y").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_jerk_y"), pl.col("original_length_ts")
            ).map_elements(
                lambda s: interpolate(s["tracing_jerk_y"], s["original_length_ts"]),
                return_dtype=pl.List(pl.Float64),
            )
        )
        .alias("tracing_jerk_y")
    )

    participants_ = participants_.with_columns(
        pl.when(
            pl.col("tracing_jerk_magnitude").is_not_null()
            & (pl.col("tracing_jerk_magnitude").list.drop_nulls().list.len() > 0)
        )
        .then(
            pl.struct(
                pl.col("tracing_jerk_magnitude"), pl.col("original_length_ts")
            ).map_elements(
                lambda s: interpolate(
                    s["tracing_jerk_magnitude"], s["original_length_ts"]
                ),
                return_dtype=pl.List(pl.Float64),
            )
        )
        .alias("tracing_jerk_magnitude")
    )

    kinematic_vars = [
        "tracing_speed",
        "tracing_speed_x",
        "tracing_speed_y",
        "tracing_speed_magnitude",
        "tracing_acceleration",
        "tracing_acceleration_x",
        "tracing_acceleration_y",
        "tracing_acceleration_magnitude",
        "tracing_jerk",
        "tracing_jerk_x",
        "tracing_jerk_y",
        "tracing_jerk_magnitude",
    ]

    for var in kinematic_vars:
        participants_ = participants_.with_columns(
            pl.when(
                pl.col(var).is_not_null()
                & (pl.col(var).list.drop_nulls().list.len() > 0)
            )
            .then(
                pl.col(var).map_elements(
                    lambda signal: _smooth_signal(
                        np.array(signal), sfreq=sfreq, moving_avg_window_ms=50
                    ),
                    return_dtype=pl.List(pl.Float64),
                )
            )
            .alias(var)
        )

    return participants_


def apply_car(
    participants: pl.DataFrame, channels: list[str], config: Config
) -> pl.DataFrame:
    participants_ = participants.with_columns(
        pl.struct(channels)
        .map_elements(
            lambda s: [sum(x) / len(x) for x in zip(*s.values())],
            return_dtype=pl.List(pl.Float64),
        )
        .over(["participant_id", "session", "block"])
        .alias("car_reference")
    )

    participants_ = participants_.with_columns(
        *[
            pl.struct([pl.col(ch), pl.col("car_reference")])
            .map_elements(
                lambda s, channel_name=ch: [
                    float(ch_val - car_val) * float(config.ieeg_process.scale_factor)
                    for ch_val, car_val in zip(s[channel_name], s["car_reference"])
                ],
                return_dtype=pl.List(pl.Float64),
            )
            .alias(ch)
            for ch in channels
        ]
    )

    participants_ = participants_.drop("car_reference")

    return participants_
