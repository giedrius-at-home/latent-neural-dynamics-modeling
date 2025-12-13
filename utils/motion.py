import numpy as np
from scipy.interpolate import interp1d
from utils.miscellaneous import contains_nulls
from typing import Optional
from scipy.signal import savgol_filter


def interpolate(behavior: list, original_length_ts: int) -> list:
    if behavior is None:
        return None
    if contains_nulls(behavior):
        return None
    if original_length_ts <= 0:
        return None

    original_indices = np.linspace(0, 1, num=len(behavior))
    target_indices = np.linspace(0, 1, num=original_length_ts)

    interpolator = interp1d(
        original_indices, behavior, kind="linear", fill_value="extrapolate"
    )
    interpolated_behavior = interpolator(target_indices)

    return interpolated_behavior.tolist()


def _smooth_signal(
    instantaneous_signal: np.ndarray, sfreq: float, moving_avg_window_ms: int = 50
) -> list:
    window_size_samples = int((moving_avg_window_ms / 1000) * sfreq)
    if window_size_samples % 2 == 0:
        window_size_samples += 1

    window_size_samples = max(3, min(window_size_samples, len(instantaneous_signal)))

    polyorder = min(3, window_size_samples - 1)

    try:
        smoothed = savgol_filter(
            instantaneous_signal,
            window_length=window_size_samples,
            polyorder=polyorder,
        )
    except Exception:
        smoothed = np.convolve(
            instantaneous_signal,
            np.ones(window_size_samples) / window_size_samples,
            mode="same",
        )

    return smoothed.tolist()


def compute_tracing_speeds(
    x: Optional[list],
    y: Optional[list],
    time: list,
) -> dict:

    result = {
        "tracing_speed": None,
        "tracing_speed_x": None,
        "tracing_speed_y": None,
        "tracing_speed_magnitude": None,
    }

    if contains_nulls(time):
        return result

    has_x = x is not None and not contains_nulls(x)
    has_y = y is not None and not contains_nulls(y)

    if not has_x and not has_y:
        return result

    dt = np.diff(time)
    dt[dt == 0] = np.finfo(float).eps

    if has_x:
        dx = np.diff(x)
        instantaneous_speed_x = dx / dt
        instantaneous_speed_x = np.insert(instantaneous_speed_x, 0, 0)
        result["tracing_speed_x"] = instantaneous_speed_x.tolist()

    if has_y:
        dy = np.diff(y)
        instantaneous_speed_y = dy / dt
        instantaneous_speed_y = np.insert(instantaneous_speed_y, 0, 0)
        result["tracing_speed_y"] = instantaneous_speed_y.tolist()

    if has_x and has_y:
        dx = np.diff(x)
        dy = np.diff(y)
        instantaneous_speed_combined = np.sqrt(dx**2 + dy**2) / dt
        instantaneous_speed_combined = np.insert(instantaneous_speed_combined, 0, 0)
        result["tracing_speed"] = instantaneous_speed_combined.tolist()

        speed_x = np.array(result["tracing_speed_x"])
        speed_y = np.array(result["tracing_speed_y"])
        speed_magnitude = np.sqrt(speed_x**2 + speed_y**2)
        result["tracing_speed_magnitude"] = speed_magnitude.tolist()

    return result


def compute_tracing_acceleration(
    speed_x: Optional[list],
    speed_y: Optional[list],
    time: list,
) -> dict:

    result = {
        "tracing_acceleration": None,
        "tracing_acceleration_x": None,
        "tracing_acceleration_y": None,
        "tracing_acceleration_magnitude": None,
    }

    if contains_nulls(time):
        return result

    has_speed_x = speed_x is not None and not contains_nulls(speed_x)
    has_speed_y = speed_y is not None and not contains_nulls(speed_y)

    if not has_speed_x and not has_speed_y:
        return result

    dt = np.diff(time)
    dt[dt == 0] = np.finfo(float).eps

    if has_speed_x:
        d_speed_x = np.diff(speed_x)
        instantaneous_acceleration_x = d_speed_x / dt
        instantaneous_acceleration_x = np.insert(instantaneous_acceleration_x, 0, 0)
        result["tracing_acceleration_x"] = instantaneous_acceleration_x.tolist()

    if has_speed_y:
        d_speed_y = np.diff(speed_y)
        instantaneous_acceleration_y = d_speed_y / dt
        instantaneous_acceleration_y = np.insert(instantaneous_acceleration_y, 0, 0)
        result["tracing_acceleration_y"] = instantaneous_acceleration_y.tolist()

    if has_speed_x and has_speed_y:
        d_speed_x = np.diff(speed_x)
        d_speed_y = np.diff(speed_y)
        instantaneous_acceleration_combined = np.sqrt(d_speed_x**2 + d_speed_y**2) / dt
        instantaneous_acceleration_combined = np.insert(
            instantaneous_acceleration_combined, 0, 0
        )
        result["tracing_acceleration"] = instantaneous_acceleration_combined.tolist()

        accel_x = np.array(result["tracing_acceleration_x"])
        accel_y = np.array(result["tracing_acceleration_y"])
        acceleration_magnitude = np.sqrt(accel_x**2 + accel_y**2)
        result["tracing_acceleration_magnitude"] = acceleration_magnitude.tolist()

    return result


def compute_tracing_jerk(
    acceleration_x: Optional[list],
    acceleration_y: Optional[list],
    time: list,
) -> dict:

    result = {
        "tracing_jerk": None,
        "tracing_jerk_x": None,
        "tracing_jerk_y": None,
        "tracing_jerk_magnitude": None,
    }

    if contains_nulls(time):
        return result

    has_accel_x = acceleration_x is not None and not contains_nulls(acceleration_x)
    has_accel_y = acceleration_y is not None and not contains_nulls(acceleration_y)

    if not has_accel_x and not has_accel_y:
        return result

    dt = np.diff(time)
    dt[dt == 0] = np.finfo(float).eps

    if has_accel_x:
        d_acceleration_x = np.diff(acceleration_x)
        instantaneous_jerk_x = d_acceleration_x / dt
        instantaneous_jerk_x = np.insert(instantaneous_jerk_x, 0, 0)
        result["tracing_jerk_x"] = instantaneous_jerk_x.tolist()

    if has_accel_y:
        d_acceleration_y = np.diff(acceleration_y)
        instantaneous_jerk_y = d_acceleration_y / dt
        instantaneous_jerk_y = np.insert(instantaneous_jerk_y, 0, 0)
        result["tracing_jerk_y"] = instantaneous_jerk_y.tolist()

    if has_accel_x and has_accel_y:
        d_acceleration_x = np.diff(acceleration_x)
        d_acceleration_y = np.diff(acceleration_y)
        instantaneous_jerk_combined = (
            np.sqrt(d_acceleration_x**2 + d_acceleration_y**2) / dt
        )
        instantaneous_jerk_combined = np.insert(instantaneous_jerk_combined, 0, 0)
        result["tracing_jerk"] = instantaneous_jerk_combined.tolist()

        jerk_x = np.array(result["tracing_jerk_x"])
        jerk_y = np.array(result["tracing_jerk_y"])
        jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2)
        result["tracing_jerk_magnitude"] = jerk_magnitude.tolist()

    return result
