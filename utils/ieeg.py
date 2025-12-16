from pathlib import Path
import mne
import numpy as np
from utils.logger import get_logger
from scipy.signal import hilbert


def preprocess_ieeg(
    ieeg_headers_file: str,
    sfreq: int,
    low_freq: int,
    high_freq: int,
    notch_freqs: list[int],
) -> dict[str, list[float]] | None:
    ieeg_path = Path(ieeg_headers_file)

    if not ieeg_path.exists():
        return None

    try:
        raw = mne.io.read_raw_brainvision(ieeg_path, preload=True, verbose=False)

        raw.notch_filter(freqs=notch_freqs, verbose=False)
        raw.filter(l_freq=low_freq, h_freq=high_freq)
        raw.resample(sfreq=sfreq, verbose=False)

        data = raw.get_data()
        channels = raw.ch_names
        channels_data = {f"art_{ch}": d.tolist() for ch, d in zip(raw.ch_names, data)}
        channels_data["sfreq"] = float(sfreq)

        return channels_data
    except Exception as e:
        from utils.logger import get_logger

        logger = get_logger()

        logger.info(str(e))
        return None


def filter_recording(
    recording: list[float],
    low_freq: int,
    high_freq: int,
    notch_freqs: list[int],
    sfreq: int,
) -> list[float]:
    recording_ = np.array(recording, dtype=np.float64)

    recording_ = mne.filter.filter_data(
        data=recording_, sfreq=sfreq, l_freq=low_freq, h_freq=high_freq, verbose=False
    )
    if notch_freqs:
        recording_ = mne.filter.notch_filter(
            x=recording_, Fs=sfreq, freqs=notch_freqs, verbose=False
        )

    return list(recording_)


def process_and_resample(
    recording: list[float],
    low_freq: float,
    high_freq: float,
    notch_freqs: list[int],
    original_sfreq: int,
    target_sfreq: int,
) -> list[float]:

    data = np.array(recording, dtype=np.float64)
    if notch_freqs:
        data = mne.filter.notch_filter(data, Fs=original_sfreq, freqs=notch_freqs, verbose=False)
    data = mne.filter.filter_data(data, sfreq=original_sfreq, l_freq=low_freq, h_freq=high_freq, verbose=False)
    data = mne.filter.resample(data, down=original_sfreq / target_sfreq, verbose=False)
    return data.tolist()


def process_dual_band(
    recording: list[float],
    low_freq: float,
    envelope_cutoff: float,
    high_freq: float,
    notch_freqs: list[int],
    original_sfreq: int,
    target_sfreq: int,
    scale_factor: float = 1.0,
) -> tuple[list[float], list[float]]:
    data = np.array(recording, dtype=np.float64)
    if notch_freqs:
        data = mne.filter.notch_filter(data, Fs=original_sfreq, freqs=notch_freqs, verbose=False)

    low_band = mne.filter.filter_data(data, sfreq=original_sfreq, l_freq=low_freq, h_freq=envelope_cutoff, verbose=False)
    low_band = mne.filter.resample(low_band, down=original_sfreq / target_sfreq, verbose=False)

    high_band = mne.filter.filter_data(data, sfreq=original_sfreq, l_freq=envelope_cutoff, h_freq=high_freq, verbose=False)
    high_band_envelope = np.abs(hilbert(high_band))
    
    window_size = int(0.2 * original_sfreq)
    kernel = np.ones(window_size) / window_size
    high_band_envelope = np.convolve(high_band_envelope, kernel, mode='same')
    
    high_band_envelope = mne.filter.resample(high_band_envelope, down=original_sfreq / target_sfreq, verbose=False)
    
    low_band = low_band * scale_factor
    high_band_envelope = high_band_envelope * scale_factor

    return low_band.tolist(), high_band_envelope.tolist()


def epoch_trials(
    recording: list[float],
    window_size: int = 2000,
    step_size: int = 500,
) -> list[list[float]]:
    n_samples = len(recording)
    recording_ = np.array(recording, dtype=np.float64)

    epochs = []
    for start in range(0, n_samples, step_size):
        end = start + window_size
        if end > n_samples:
            last_epoch = recording_[start:]
            padding = np.zeros(end - n_samples)
            epoch = np.concatenate([last_epoch, padding])
        else:
            epoch = recording_[start:end]
        epochs.append(epoch.tolist())

    return epochs


def calculate_psd_welch(
    epochs: list[list[float]],
    sfreq: int = 1000,
    low_freq: float = 3.0,
    high_freq: float = 250.0,
) -> list[list]:
    epochs_arr = np.array([np.asarray(epoch, dtype=np.float64) for epoch in epochs])
    psds, freqs = mne.time_frequency.psd_array_welch(
        epochs_arr,
        sfreq=sfreq,
        fmin=low_freq,
        fmax=high_freq,
        average="mean",
        n_fft=sfreq,
        n_jobs=-1,
        verbose=False,
    )
    return [freqs.tolist(), psds.tolist()]
