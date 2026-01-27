import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from utils.miscellaneous import contains_nulls
from typing import Optional


SAVGOL_WINDOW_LENGTH = 31
SAVGOL_POLYORDER = 4


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


def savgol_derivative(
    signal: list,
    sfreq: float,
    deriv: int = 0,
    window_length: int = SAVGOL_WINDOW_LENGTH,
    polyorder: int = SAVGOL_POLYORDER,
) -> Optional[list]:
    if signal is None or contains_nulls(signal):
        return None
    if len(signal) < window_length:
        return None

    arr = np.array(signal, dtype=np.float64)
    delta = 1.0 / sfreq if deriv > 0 else 1.0

    result = savgol_filter(
        arr,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
    )
    return result.tolist()


def compute_magnitude(comp_x: list, comp_y: list) -> Optional[list]:
    if comp_x is None or comp_y is None:
        return None
    if contains_nulls(comp_x) or contains_nulls(comp_y):
        return None

    x = np.array(comp_x, dtype=np.float64)
    y = np.array(comp_y, dtype=np.float64)
    return np.sqrt(x**2 + y**2).tolist()


def compute_angle(comp_x: list, comp_y: list) -> Optional[list]:
    if comp_x is None or comp_y is None:
        return None
    if contains_nulls(comp_x) or contains_nulls(comp_y):
        return None

    x = np.array(comp_x, dtype=np.float64)
    y = np.array(comp_y, dtype=np.float64)
    return np.arctan2(y, x).tolist()


def compute_cos(angle: list) -> Optional[list]:
    if angle is None or contains_nulls(angle):
        return None
    return np.cos(np.array(angle, dtype=np.float64)).tolist()


def compute_sin(angle: list) -> Optional[list]:
    if angle is None or contains_nulls(angle):
        return None
    return np.sin(np.array(angle, dtype=np.float64)).tolist()
