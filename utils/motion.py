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


def compute_tracing_speeds(
    x: Optional[list],
    y: Optional[list],
    time: list,
    moving_avg_window_ms: int = 50,
) -> dict:

    result = {"combined": None, "x": None, "y": None}

    if contains_nulls(time):
        return result

    has_x = x is not None and not contains_nulls(x)
    has_y = y is not None and not contains_nulls(y)

    if not has_x and not has_y:
        return result

    dt = np.diff(time)
    dt[dt == 0] = np.finfo(float).eps

    def smooth_speed(instantaneous_speed: np.ndarray) -> list:
        window_size_samples = int((moving_avg_window_ms / 1000) * 1000)
        if window_size_samples % 2 == 0:
            window_size_samples += 1

        window_size_samples = max(3, min(window_size_samples, len(instantaneous_speed)))

        polyorder = min(3, window_size_samples - 1)

        try:
            smoothed = savgol_filter(
                instantaneous_speed,
                window_length=window_size_samples,
                polyorder=polyorder,
            )
        except Exception:
            smoothed = np.convolve(
                instantaneous_speed,
                np.ones(window_size_samples) / window_size_samples,
                mode="same",
            )

        return smoothed.tolist()

    if has_x:
        dx = np.diff(x)
        instantaneous_speed_x = dx / dt
        instantaneous_speed_x = np.insert(instantaneous_speed_x, 0, 0)
        result["x"] = smooth_speed(instantaneous_speed_x)

    if has_y:
        dy = np.diff(y)
        instantaneous_speed_y = dy / dt
        instantaneous_speed_y = np.insert(instantaneous_speed_y, 0, 0)
        result["y"] = smooth_speed(instantaneous_speed_y)

    if has_x and has_y:
        dx = np.diff(x) if has_x else np.zeros(len(time) - 1)
        dy = np.diff(y) if has_y else np.zeros(len(time) - 1)
        instantaneous_speed_combined = np.sqrt(dx**2 + dy**2) / dt
        instantaneous_speed_combined = np.insert(instantaneous_speed_combined, 0, 0)
        result["combined"] = smooth_speed(instantaneous_speed_combined)

    return result
