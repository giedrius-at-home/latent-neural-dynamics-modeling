import numpy as np
from scipy.interpolate import interp1d


def find_overlap_window(neural_time, behavior_time):
    overlap_start = max(neural_time[0], behavior_time[0])
    overlap_end = min(neural_time[-1], behavior_time[-1])
    return overlap_start, overlap_end


def generate_master_grid(start_time, end_time, target_sfreq=60.0):
    n_samples = int((end_time - start_time) * target_sfreq) + 1
    return np.linspace(
        start_time, start_time + (n_samples - 1) / target_sfreq, n_samples
    )


def interpolate_to_grid(signal, original_time, master_time):
    interpolator = interp1d(
        original_time, signal, kind="linear", bounds_error=False, fill_value=np.nan
    )
    return interpolator(master_time)


def trim_to_overlap(signal, time, overlap_start, overlap_end):
    mask = (time >= overlap_start) & (time <= overlap_end)
    return signal[mask], time[mask]
