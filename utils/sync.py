import numpy as np
from scipy.interpolate import interp1d


def find_overlap_window(neural_time, behavior_time):
    overlap_start = max(neural_time[0], behavior_time[0])
    overlap_end = min(neural_time[-1], behavior_time[-1])
    return overlap_start, overlap_end


def generate_master_grid(start_time, end_time, target_sfreq=80.0):
    n_samples = int((end_time - start_time) * target_sfreq)
    return np.linspace(
        start_time, start_time + (n_samples - 1) / target_sfreq, n_samples
    )


def interpolate_to_grid(signal, original_time, master_time):
    """Linear interp of ``signal`` (defined at ``original_time``) onto
    ``master_time``. Out-of-bounds master samples → edge-fill with
    ``signal[0]`` / ``signal[-1]``. Edge-fill (vs ``np.nan``) keeps the
    output finite even when ``master_time`` overruns ``original_time`` by a
    sample or two due to rounding in the resample grid construction.
    """
    if len(signal) == 0 or len(original_time) == 0:
        return np.full(len(master_time), np.nan, dtype=np.float64)
    interpolator = interp1d(
        original_time,
        signal,
        kind="linear",
        bounds_error=False,
        fill_value=(float(signal[0]), float(signal[-1])),
    )
    return interpolator(master_time)


def trim_to_overlap(signal, time, overlap_start, overlap_end):
    mask = (time >= overlap_start) & (time <= overlap_end)
    return signal[mask], time[mask]
