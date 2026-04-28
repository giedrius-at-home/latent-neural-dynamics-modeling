import polars as pl
import numpy as np
from pathlib import Path

from utils.ieeg import epoch_trials, calculate_psd_welch
from utils.polars import band_pass_resample, read_and_implode_parquet
from .motion import construct_motion_table
from utils.file_handling import get_child_subchilds_tuples
from utils.config import Config
from utils.logger import get_logger
from utils.motion import (
    savgol_derivative,
    compute_magnitude,
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

    max_pause_seconds = getattr(config.ieeg_process, "max_pause_seconds", None)

    participants = _chunk_recordings(
        participants,
        all_band_channels,
        config.ieeg_process.chunk_margin,
        config.ieeg_process.resampled_freq,
        max_pause_seconds=max_pause_seconds,
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


def construct_participants_table(
    config: Config, participant_id: str = None, session: str = None
):
    logger = get_logger()
    data_path = Path(config.data_directory)
    save_path = Path(config.save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    participants_partitions = get_child_subchilds_tuples(
        data_path / config.participants_intermediate_table_name
    )

    for p_part in participants_partitions:
        root, p_id, s_id, block = p_part

        # Extract values if in key=value format
        p_val = p_id.split("=")[-1] if "=" in p_id else p_id
        s_val = s_id.split("=")[-1] if "=" in s_id else s_id

        # Filtering logic
        if participant_id is not None and p_val != participant_id:
            continue
        if session is not None and str(s_val) != str(session):
            continue

        p_partition_path = data_path / root / p_id / s_id / block / "*"
        participants = pl.read_parquet(p_partition_path)
        logger.info(f"Loaded participants from: {p_partition_path}")

        if "is_fragmented" in participants.columns:
            n_before = participants.height
            participants = participants.filter(pl.col("is_fragmented") == False)
            n_dropped = n_before - participants.height
            if n_dropped > 0:
                logger.info(f"Filtered {n_dropped} fragmented block(s)")
            participants = participants.drop("is_fragmented")

        if participants.is_empty():
            logger.info(f"Skipping empty partition after fragmented filter: {p_partition_path}")
            continue

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
            "x",
            "y",
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


def _compute_laplacian(lfp_k, lfp_k1, lfp_k2):
    """Compute Laplacian: D_k(t) = (LFP_k(t) - LFP_{k+1}(t)) - (LFP_{k+1}(t) - LFP_{k+2}(t))
    Simplified: D_k(t) = LFP_k(t) - 2*LFP_{k+1}(t) + LFP_{k+2}(t)
    """
    if lfp_k is None or lfp_k1 is None or lfp_k2 is None:
        return None
    if len(lfp_k) == 0 or len(lfp_k1) == 0 or len(lfp_k2) == 0:
        return None

    lfp_k_arr = np.array(lfp_k, dtype=np.float64)
    lfp_k1_arr = np.array(lfp_k1, dtype=np.float64)
    lfp_k2_arr = np.array(lfp_k2, dtype=np.float64)

    # Ensure all arrays have the same length
    min_len = min(len(lfp_k_arr), len(lfp_k1_arr), len(lfp_k2_arr))
    lfp_k_arr = lfp_k_arr[:min_len]
    lfp_k1_arr = lfp_k1_arr[:min_len]
    lfp_k2_arr = lfp_k2_arr[:min_len]

    laplacian = lfp_k_arr - 2 * lfp_k1_arr + lfp_k2_arr
    return laplacian.tolist()


def _add_ieeg_data(
    participants: pl.DataFrame, config: Config
) -> tuple[pl.DataFrame, list[str]]:
    from utils.ieeg import process_all_bands

    logger = get_logger()

    # Load iEEG data
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

    drop_lfp = getattr(config.ieeg_process, "drop_lfp", False)
    if drop_lfp:
        lfp_cols_to_drop = [ch for ch in LFP_CHANNELS if ch in participants_.columns]
        if lfp_cols_to_drop:
            participants_ = participants_.drop(lfp_cols_to_drop)
            logger.info(
                f"Dropped {len(lfp_cols_to_drop)} LFP channels early (drop_lfp=True)"
            )

    original_sfreq = 1000
    target_sfreq = config.ieeg_process.resampled_freq
    raw_bands = getattr(config.ieeg_process, "raw_bands", {})
    envelope_bands = getattr(config.ieeg_process, "envelope_bands", {})
    log_power_bands = getattr(config.ieeg_process, "log_power_bands", {})
    notch_freqs = config.ieeg_process.notch_freqs
    scale_factor = float(getattr(config.ieeg_process, "scale_factor", 1.0))
    compute_average = getattr(config.ieeg_process, "compute_average", False)
    apply_car = getattr(config.ieeg_process, "apply_car", False)

    all_band_names = (
        list(raw_bands.keys())
        + list(envelope_bands.keys())
        + list(log_power_bands.keys())
    )
    all_band_channels = []

    # Common Average Reference across ECOG channels (at raw 1000 Hz, pre-filter).
    # Subtracts the per-timestep mean of ECOG_1..ECOG_4 from each ECOG channel.
    if apply_car:
        ecog_present = [ch for ch in ECOG_CHANNELS if ch in participants_.columns]
        if len(ecog_present) >= 2:
            logger.info(f"Applying CAR across {len(ecog_present)} ECOG channels")

            def _ecog_car_mean(row, cols=ecog_present):
                signals = []
                for c in cols:
                    s = row[c]
                    if s is not None and len(s) > 0:
                        signals.append(np.asarray(s, dtype=np.float64))
                if not signals:
                    return []
                min_len = min(len(s) for s in signals)
                stacked = np.stack([s[:min_len] for s in signals], axis=0)
                return stacked.mean(axis=0).tolist()

            participants_ = participants_.with_columns(
                pl.struct(ecog_present)
                .map_elements(_ecog_car_mean, return_dtype=pl.List(pl.Float64))
                .alias("_ecog_car_mean")
            )

            for ch in ecog_present:
                def _subtract_mean(row, ch_name=ch):
                    s = row[ch_name]
                    m = row["_ecog_car_mean"]
                    if s is None or len(s) == 0:
                        return s
                    if m is None or len(m) == 0:
                        return s
                    s_arr = np.asarray(s, dtype=np.float64)
                    m_arr = np.asarray(m, dtype=np.float64)
                    n = min(len(s_arr), len(m_arr))
                    return (s_arr[:n] - m_arr[:n]).tolist()

                participants_ = participants_.with_columns(
                    pl.struct([ch, "_ecog_car_mean"])
                    .map_elements(_subtract_mean, return_dtype=pl.List(pl.Float32))
                    .alias(ch)
                )

            participants_ = participants_.drop("_ecog_car_mean")
        else:
            logger.info("apply_car=True but fewer than 2 ECOG channels present; skipping CAR")

    # Process ECOG channels (ECOG_1 to ECOG_4)
    logger.info("Processing ECOG channels")
    for ecog_ch in ECOG_CHANNELS:
        if ecog_ch not in participants_.columns:
            continue

        def _process_bands(r, ch_name=ecog_ch):
            if r is None or len(r) == 0:
                return {band: [] for band in all_band_names}
            return process_all_bands(
                r,
                raw_bands=raw_bands,
                envelope_bands=envelope_bands,
                log_power_bands=log_power_bands,
                notch_freqs=notch_freqs,
                original_sfreq=original_sfreq,
                target_sfreq=target_sfreq,
                scale_factor=scale_factor,
            )

        participants_ = participants_.with_columns(
            pl.col(ecog_ch)
            .map_elements(_process_bands, return_dtype=pl.Object)
            .alias(f"_bands_{ecog_ch}")
        )

        for band_name in all_band_names:
            band_channel = f"{ecog_ch}_{band_name}"
            all_band_channels.append(band_channel)
            participants_ = participants_.with_columns(
                pl.col(f"_bands_{ecog_ch}")
                .map_elements(
                    lambda x, bn=band_name: x[bn], return_dtype=pl.List(pl.Float32)
                )
                .alias(band_channel)
            )

        participants_ = participants_.drop(f"_bands_{ecog_ch}", ecog_ch)

    # Compute average across ECOG channels for each band if requested
    if compute_average:
        logger.info("Computing average across ECOG channels for each band")
        for band_name in all_band_names:
            # Collect all ECOG band channels for this band
            ecog_band_cols = [
                f"{ecog_ch}_{band_name}"
                for ecog_ch in ECOG_CHANNELS
                if f"{ecog_ch}_{band_name}" in participants_.columns
            ]

            if len(ecog_band_cols) > 0:
                # Compute average across all ECOG channels for this band
                avg_channel = f"ECOG_avg_{band_name}"
                all_band_channels.append(avg_channel)

                def _compute_ecog_avg(row, cols=ecog_band_cols):
                    """Compute average across ECOG channels for a single band"""
                    valid_signals = []
                    for col in cols:
                        signal = row[col]
                        if signal is not None and len(signal) > 0:
                            valid_signals.append(np.array(signal, dtype=np.float32))

                    if len(valid_signals) == 0:
                        return []

                    # Ensure all signals have the same length
                    min_len = min(len(s) for s in valid_signals)
                    valid_signals = [s[:min_len] for s in valid_signals]

                    # Compute average
                    avg_signal = np.mean(valid_signals, axis=0)
                    return avg_signal.tolist()

                participants_ = participants_.with_columns(
                    pl.struct(ecog_band_cols)
                    .map_elements(_compute_ecog_avg, return_dtype=pl.List(pl.Float32))
                    .alias(avg_channel)
                )

    # Process LFP channels: Compute Laplacian for k ∈ {9, 10, 11, 12, 13, 14}
    # To double the number, we'll also include k ∈ {8, 15} if available
    if not drop_lfp:
        logger.info("Processing LFP channels with Laplacian")
        laplacian_k_values = [
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ]  # Doubled from original 6 to 8

        for k in laplacian_k_values:
            lfp_k_name = f"LFP_{k}"
            lfp_k1_name = f"LFP_{k+1}"
            lfp_k2_name = f"LFP_{k+2}"

            # Check if all required channels exist
            if not all(
                ch in participants_.columns
                for ch in [lfp_k_name, lfp_k1_name, lfp_k2_name]
            ):
                continue

            # Compute Laplacian using Polars expressions
            # D_k(t) = LFP_k(t) - 2*LFP_{k+1}(t) + LFP_{k+2}(t)
            def _laplacian_wrapper(
                s, k_name=lfp_k_name, k1_name=lfp_k1_name, k2_name=lfp_k2_name
            ):
                if s is None:
                    return None
                return _compute_laplacian(s[k_name], s[k1_name], s[k2_name])

            participants_ = participants_.with_columns(
                pl.struct([lfp_k_name, lfp_k1_name, lfp_k2_name])
                .map_elements(_laplacian_wrapper, return_dtype=pl.List(pl.Float32))
                .alias(f"_laplacian_{k}")
            )

            # Process Laplacian signal through bands
            def _process_bands_laplacian(r):
                if r is None or len(r) == 0:
                    return {band: [] for band in all_band_names}
                return process_all_bands(
                    r,
                    raw_bands=raw_bands,
                    envelope_bands=envelope_bands,
                    log_power_bands=log_power_bands,
                    notch_freqs=notch_freqs,
                    original_sfreq=original_sfreq,
                    target_sfreq=target_sfreq,
                    scale_factor=scale_factor,
                )

            participants_ = participants_.with_columns(
                pl.col(f"_laplacian_{k}")
                .map_elements(_process_bands_laplacian, return_dtype=pl.Object)
                .alias(f"_bands_laplacian_{k}")
            )

            for band_name in all_band_names:
                # Split band_name into band and feature (e.g., "delta_raw" -> band="delta", feature="raw")
                # Handle different formats: "delta_raw", "lowbeta_env", "raw" (no underscore)
                if "_" in band_name:
                    parts = band_name.rsplit("_", 1)
                    if len(parts) == 2:
                        band, feature = parts
                    else:
                        band = band_name
                        feature = "raw"
                else:
                    band = band_name
                    feature = "raw"

                # Create channel name: LAPLACIAN_k-{k+2}_LFP_{band}_{feature}
                laplacian_channel = f"LAPLACIAN_{k}-{k+2}_LFP_{band}_{feature}"
                all_band_channels.append(laplacian_channel)
                participants_ = participants_.with_columns(
                    pl.col(f"_bands_laplacian_{k}")
                    .map_elements(
                        lambda x, bn=band_name: x[bn], return_dtype=pl.List(pl.Float32)
                    )
                    .alias(laplacian_channel)
                )

            participants_ = participants_.drop(
                f"_laplacian_{k}", f"_bands_laplacian_{k}"
            )

        # Drop all original LFP channels (if they weren't already dropped)
        lfp_cols_to_drop = [ch for ch in LFP_CHANNELS if ch in participants_.columns]
        if lfp_cols_to_drop:
            participants_ = participants_.drop(lfp_cols_to_drop)
            logger.info(f"Dropped {len(lfp_cols_to_drop)} original LFP channels")
    else:
        logger.info("Skipping LFP processing (drop_lfp=True)")

    return participants_, all_band_channels


def _longest_stationary_run_seconds(x_list, y_list, trial_time):
    """Return the duration (in seconds) of the longest run of identical (x,y) positions."""
    if x_list is None or y_list is None or trial_time is None:
        return None
    x = np.array(x_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    n = len(x)
    if n < 2 or trial_time <= 0:
        return 0.0
    dt = trial_time / (n - 1)
    max_run = 1
    cur_run = 1
    for j in range(1, n):
        if x[j] == x[j - 1] and y[j] == y[j - 1]:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 1
    return float(max_run * dt)


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
    max_pause_seconds: float = None,
) -> pl.DataFrame:
    logger = get_logger()

    participants_ = (
        participants.with_columns(
            (pl.col("onset") - chunk_margin).alias("margined_onset"),
            (pl.col("trial_time") + 2 * chunk_margin).alias("margined_duration"),
        )
        .with_columns(
            (pl.col("margined_onset") * sfreq)
            .round()
            .cast(pl.UInt32)
            .alias("start_ts"),
            (pl.col("margined_duration") * sfreq)
            .round()
            .cast(pl.UInt32)
            .alias("chunk_length_ts"),
            (pl.col("trial_time") * sfreq)
            .round()
            .cast(pl.UInt32)
            .alias("original_length_ts"),
            (pl.col("onset") * sfreq).round().cast(pl.UInt32).alias("onset_ts"),
        )
        .with_columns(
            (
                pl.int_ranges(0, pl.col("chunk_length_ts"), dtype=pl.UInt32)
                + pl.col("start_ts")
            )
            .truediv(sfreq)
            .alias("time"),
            (
                pl.int_ranges(0, pl.col("original_length_ts"), dtype=pl.UInt32)
                + pl.col("onset_ts")
            )
            .truediv(sfreq)
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

    if max_pause_seconds is not None and max_pause_seconds > 0:
        participants_ = participants_.with_columns(
            pl.struct("x", "y", "trial_time")
            .map_elements(
                lambda s: _longest_stationary_run_seconds(
                    s["x"], s["y"], s["trial_time"]
                ),
                return_dtype=pl.Float64,
            )
            .alias("_max_pause_s")
        )
        n_before = participants_.height
        participants_ = participants_.filter(
            pl.col("_max_pause_s") <= max_pause_seconds
        ).drop("_max_pause_s")
        n_dropped = n_before - participants_.height
        if n_dropped > 0:
            logger.info(
                f"Filtered {n_dropped} trial(s) with stationary pause > {max_pause_seconds}s"
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
