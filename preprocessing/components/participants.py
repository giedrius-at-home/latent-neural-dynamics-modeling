import polars as pl
import numpy as np
from pathlib import Path

from utils.ieeg import epoch_trials, calculate_psd_welch
from utils.polars import band_pass_resample
from .motion import construct_motion_table
from utils.file_handling import get_child_subchilds_tuples
from utils.config import Config
from utils.logger import get_logger
from utils.motion import (
    savgol_derivative,
    compute_magnitude,
    compute_angle,
    compute_cos,
    compute_sin,
)
from utils.sync import interpolate_to_grid

LFP_SCHEMA = pl.Struct(
    [pl.Field(f"LFP_{i}", pl.List(pl.Float32)) for i in range(1, 17)]
)

ECOG_SCHEMA = pl.Struct(
    [pl.Field(f"ECOG_{i}", pl.List(pl.Float32)) for i in range(1, 5)]
)

EOG_SCHEMA = pl.Struct([pl.Field(f"EOG_{i}", pl.List(pl.Float32)) for i in range(1, 5)])

iEEG_SCHEMA = pl.Struct(
    [
        *LFP_SCHEMA.fields,
        *ECOG_SCHEMA.fields,
        *EOG_SCHEMA.fields,
        pl.Field("sfreq", pl.List(pl.Float32)),
    ]
)

LFP_CHANNELS = [f"LFP_{i}" for i in range(1, 17)]
ECOG_CHANNELS = [f"ECOG_{i}" for i in range(1, 5)]

KINEMATIC_COMPONENTS = ["x", "y"]
KINEMATIC_DERIVATIVES = {
    "velocity": 1,
    "acceleration": 2,
    "jerk": 3,
}


def _add_full_data(
    participants: pl.DataFrame, config: Config
) -> tuple[pl.DataFrame, list[str]]:
    logger = get_logger()
    logger.info("Adding full data")

    ieeg_participants, all_band_channels = _add_ieeg_data(participants, config)
    logger.info(f"Loaded iEEG data with {len(all_band_channels)} band channels")

    motion_participants = construct_motion_table(ieeg_participants, config)
    logger.info("Loaded motion data")

    ieeg_participants = (
        ieeg_participants.explode("trials")
        .with_columns(
            pl.col("onsets")
            .list.get(pl.col("trials") - 1, null_on_oob=True)
            .alias("onset")
        )
        .drop("onsets", "yscores", "trial_index", strict=False)
        .with_columns(pl.col("trials").alias("trial"))
        .drop("trials")
    )

    participants = ieeg_participants.join(
        motion_participants,
        on=["participant_id", "session", "block", "trial"],
        how="left",
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
        all_band_channels,
        config.ieeg_process.chunk_margin,
        config.ieeg_process.resampled_freq,
    )

    for channel in all_band_channels:
        if channel in participants.columns:
            participants = participants.with_columns(
                pl.col(channel)
                .map_elements(epoch_trials, return_dtype=pl.List(pl.List(pl.Float64)))
                .alias(f"{channel}_epochs")
            )

    for channel in all_band_channels:
        epochs_col = f"{channel}_epochs"
        if epochs_col in participants.columns:
            participants = (
                participants.with_columns(
                    pl.col(epochs_col)
                    .map_elements(
                        lambda x: calculate_psd_welch(
                            x,
                            sfreq=config.ieeg_process.resampled_freq,
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
                    .map_elements(
                        lambda x: x[1], return_dtype=pl.List(pl.List(pl.Float64))
                    )
                    .alias(f"{channel}_psd_values"),
                )
                .drop(f"{channel}_psd")
            )

    return participants, all_band_channels


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

        participants, all_band_channels = _add_full_data(participants, config)

        base_cols = [
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
            "chunk_margin",
            "stim",
        ]

        band_cols = [
            c
            for c in participants.columns
            if any(c.startswith(ch) for ch in all_band_channels)
        ]
        kinematic_cols = [
            c
            for c in participants.columns
            if c.startswith("tracing_") or c in ["x_smooth", "y_smooth"]
        ]

        select_cols = base_cols + band_cols + kinematic_cols
        select_cols = [c for c in select_cols if c in participants.columns]

        participants = participants.select(select_cols)
        participants.write_parquet(
            save_path / config.output_participants_table_name,
            partition_by=["participant_id", "session", "block"],
        )
        logger.info(f"Saved to {save_path / config.output_participants_table_name}")


def _add_ieeg_data(
    participants: pl.DataFrame, config: Config
) -> tuple[pl.DataFrame, list[str]]:
    return band_pass_resample(participants, config, iEEG_SCHEMA)


def _interpolate_to_grid_wrapper(kinematic_signal, motion_time, time_original):
    if kinematic_signal is None or motion_time is None or time_original is None:
        return None
    if len(kinematic_signal) == 0 or len(motion_time) == 0 or len(time_original) == 0:
        return None
    signal_arr = np.array(kinematic_signal, dtype=np.float64)
    motion_arr = np.array(motion_time, dtype=np.float64)
    grid_arr = np.array(time_original, dtype=np.float64)
    interpolated = interpolate_to_grid(signal_arr, motion_arr, grid_arr)
    return interpolated.tolist()


def _chunk_recordings(
    participants: pl.DataFrame,
    all_band_channels: list[str],
    chunk_margin: int,
    sfreq: int,
) -> pl.DataFrame:
    logger = get_logger()

    participants_ = (
        participants.with_columns(
            (pl.col("onset") - chunk_margin).alias("margined_onset"),
            (pl.col("trial_time") + 2 * chunk_margin).alias("margined_duration"),
        )
        .with_columns(
            (pl.col("margined_onset") * sfreq).cast(pl.UInt32).alias("start_ts"),
            (pl.col("margined_duration") * sfreq)
            .cast(pl.UInt32)
            .alias("chunk_length_ts"),
            (pl.col("trial_time") * sfreq).cast(pl.UInt32).alias("original_length_ts"),
        )
        .with_columns(
            pl.int_ranges(0, pl.col("chunk_length_ts"), dtype=pl.UInt32)
            .truediv(sfreq)
            .add(pl.col("margined_onset"))
            .alias("time"),
            pl.int_ranges(0, pl.col("original_length_ts"), dtype=pl.UInt32)
            .truediv(sfreq)
            .add(pl.col("onset"))
            .alias("time_original"),
        )
    )

    for channel in all_band_channels:
        if channel in participants_.columns:
            participants_ = participants_.with_columns(
                pl.col(channel).list.slice(
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

    logger.info(
        f"Null motion_time rows: {participants_.filter(pl.col('motion_time').is_null()).height}"
    )

    participants_ = _compute_kinematics(participants_, sfreq)

    return participants_


def _compute_kinematics(participants: pl.DataFrame, sfreq: int) -> pl.DataFrame:
    for component in KINEMATIC_COMPONENTS:
        for deriv_name, deriv_order in KINEMATIC_DERIVATIVES.items():
            col_name = f"tracing_{deriv_name}_{component}"

            participants = participants.with_columns(
                pl.col(component)
                .map_elements(
                    lambda s, d=deriv_order, sf=sfreq: savgol_derivative(s, sf, d),
                    return_dtype=pl.List(pl.Float64),
                )
                .alias(col_name)
            )

    for deriv_name in KINEMATIC_DERIVATIVES.keys():
        x_col = f"tracing_{deriv_name}_x"
        y_col = f"tracing_{deriv_name}_y"

        participants = participants.with_columns(
            pl.struct(x_col, y_col)
            .map_elements(
                lambda s, xc=x_col, yc=y_col: compute_magnitude(s[xc], s[yc]),
                return_dtype=pl.List(pl.Float64),
            )
            .alias(f"tracing_{deriv_name}_magnitude")
        )

        participants = participants.with_columns(
            pl.struct(x_col, y_col)
            .map_elements(
                lambda s, xc=x_col, yc=y_col: compute_angle(s[xc], s[yc]),
                return_dtype=pl.List(pl.Float64),
            )
            .alias(f"tracing_{deriv_name}_angle")
        )

        participants = participants.with_columns(
            pl.col(f"tracing_{deriv_name}_angle")
            .map_elements(compute_cos, return_dtype=pl.List(pl.Float64))
            .alias(f"tracing_{deriv_name}_cos")
        )

        participants = participants.with_columns(
            pl.col(f"tracing_{deriv_name}_angle")
            .map_elements(compute_sin, return_dtype=pl.List(pl.Float64))
            .alias(f"tracing_{deriv_name}_sin")
        )

    kinematic_cols = [c for c in participants.columns if c.startswith("tracing_")]
    for col in kinematic_cols:
        participants = participants.with_columns(
            pl.struct(col, "motion_time", "time_original")
            .map_elements(
                lambda s, c=col: _interpolate_to_grid_wrapper(
                    s[c], s["motion_time"], s["time_original"]
                ),
                return_dtype=pl.List(pl.Float64),
            )
            .alias(col)
        )

    return participants
