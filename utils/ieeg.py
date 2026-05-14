import mne
import numpy as np
from scipy.signal import hilbert, spectrogram


def process_all_bands(
    recording: list[float],
    target_sfreq: int,
    raw_bands: dict[str, list[float]] = None,
    envelope_bands: dict[str, list[float]] = None,
    log_power_bands: dict[str, list[float]] = None,
    morlet_log_power_bands: dict[str, list[float]] = None,
    morlet_n_cycles_factor: float = 0.5,
    morlet_n_cycles_min: float = 3.0,
    morlet_freqs_per_band: int = 3,
    welch_log_power_bands: dict[str, list[float]] = None,
    welch_nperseg: int = 200,
    welch_noverlap: int = None,
    welch_nfft: int = 1024,
    notch_freqs: list[int] = None,
    original_sfreq: int = 1000,
    scale_factor: float = 1.0,
) -> dict[str, list[float]]:
    """Compute the band-features that are specified in the config.

    Each ``*_bands`` block is independent; only non-empty blocks contribute
    output. Modes:

    - ``raw_bands``: bandpass filter, resample, scale → narrow-band waveform.
    - ``envelope_bands``: bandpass + Hilbert envelope, smoothed and resampled.
    - ``log_power_bands``: as envelope but ``2 * log10(...)`` (legacy dB-ish).
    - ``morlet_log_power_bands``: Morlet TFR power, mean across band freqs,
      resampled, ``10 * log10`` (proper dB) — see ``compute_morlet_log_power``.
    - ``welch_log_power_bands``: Welch sliding-window PSD, band-integrated,
      ``10 * log10`` — see ``compute_welch_log_power``. Hop = original_sfreq /
      target_sfreq, so output is naturally at ``target_sfreq``.
    """
    result: dict[str, list[float]] = {}
    data = np.array(recording, dtype=np.float64)

    if notch_freqs:
        data = mne.filter.notch_filter(
            data, Fs=original_sfreq, freqs=notch_freqs, verbose=False
        )

    def _process_single_band(low, high, mode):
        filtered = mne.filter.filter_data(
            data, sfreq=original_sfreq, l_freq=low, h_freq=high, verbose=False
        )
        if mode == "raw":
            processed = filtered * scale_factor
        else:
            envelope = np.abs(hilbert(filtered))
            window_size = int(0.2 * original_sfreq)
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(envelope, kernel, mode="same")
            if mode == "envelope":
                processed = smoothed * scale_factor
            elif mode == "log_power":
                processed = 2 * np.log10(smoothed * scale_factor + 1e-12)
        resampled = mne.filter.resample(
            processed, down=original_sfreq / target_sfreq, verbose=False
        )
        return resampled.tolist()

    for bands, mode in [
        (raw_bands or {}, "raw"),
        (envelope_bands or {}, "envelope"),
        (log_power_bands or {}, "log_power"),
    ]:
        for band_name, (low, high) in bands.items():
            result[band_name] = _process_single_band(low, high, mode)

    if morlet_log_power_bands:
        morlet_results = compute_morlet_log_power(
            data,
            bands=morlet_log_power_bands,
            target_sfreq=target_sfreq,
            original_sfreq=original_sfreq,
            n_cycles_factor=morlet_n_cycles_factor,
            n_cycles_min=morlet_n_cycles_min,
            freqs_per_band=morlet_freqs_per_band,
            scale_factor=scale_factor,
        )
        result.update(morlet_results)

    if welch_log_power_bands:
        welch_results = compute_welch_log_power(
            data,
            bands=welch_log_power_bands,
            target_sfreq=target_sfreq,
            original_sfreq=original_sfreq,
            nperseg=welch_nperseg,
            noverlap=welch_noverlap,
            nfft=welch_nfft,
            scale_factor=scale_factor,
        )
        result.update(welch_results)

    return result


def compute_welch_log_power(
    recording: list[float],
    bands: dict[str, list[float]],
    target_sfreq: int,
    original_sfreq: int = 1000,
    nperseg: int = 200,
    noverlap: int = None,
    nfft: int = 1024,
    scale_factor: float = 1.0,
) -> dict[str, list[float]]:
    """Welch sliding-window log-power dB per band, output at ``target_sfreq``.

    Hop = ``original_sfreq / target_sfreq`` (e.g. 1000/20 = 50 samples), set
    via ``noverlap = nperseg - hop``. Each window: PSD via Welch (Hann taper,
    ``scaling='density'``), integrate across band freq bins (``mean``), then
    ``10 * log10`` for dB. Zero-padding (``nfft > nperseg``) gives narrower
    freq bins so even thin (~5 Hz) bands always contain ≥ 1 bin.

    Match Sani et al. PSID/DPAD preprocessing convention.
    """
    data = np.asarray(recording, dtype=np.float64)
    if data.size == 0:
        return {band: [] for band in bands}

    data = data * scale_factor

    hop = int(round(original_sfreq / target_sfreq))
    nperseg = int(nperseg)
    if nperseg > data.size:
        nperseg = data.size
    if noverlap is None:
        noverlap = max(0, nperseg - hop)
    nfft = max(int(nfft), nperseg)

    freqs, _times, Sxx = spectrogram(
        data,
        fs=original_sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="density",
        mode="psd",
        window="hann",
    )

    out: dict[str, list[float]] = {}
    for band in bands:
        lo, hi = float(bands[band][0]), float(bands[band][1])
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            out[band] = []
            continue
        bp = Sxx[mask, :].mean(axis=0)
        out[band] = (10.0 * np.log10(bp + 1e-20)).tolist()
    return out


def compute_morlet_log_power(
    recording: list[float],
    bands: dict[str, list[float]],
    target_sfreq: int,
    original_sfreq: int = 1000,
    n_cycles_factor: float = 0.5,
    n_cycles_min: float = 3.0,
    freqs_per_band: int = 3,
    scale_factor: float = 1.0,
) -> dict[str, list[float]]:
    """Morlet wavelet log-power dB per band, output at ``target_sfreq``.

    For each band: ``freqs_per_band`` linearly-spaced freqs span band edges.
    ``n_cycles = max(n_cycles_min, freq * n_cycles_factor)`` — fixed
    ``1/n_cycles_factor`` second window across freqs (with a low-freq floor
    so theta gets at least ``n_cycles_min`` cycles). Per-band power =
    ``mean(|morlet|^2)`` across band freqs, decimated 1 kHz → ``target_sfreq``
    via ``mne.filter.resample``, then ``10 * log10``.
    """
    data = np.asarray(recording, dtype=np.float64)
    if data.size == 0:
        return {band: [] for band in bands}

    data = data * scale_factor

    band_names = list(bands.keys())
    band_freqs: list[np.ndarray] = []
    for band in band_names:
        lo, hi = float(bands[band][0]), float(bands[band][1])
        band_freqs.append(np.linspace(lo, hi, freqs_per_band))
    all_freqs = np.concatenate(band_freqs)
    n_cycles = np.maximum(n_cycles_min, all_freqs * n_cycles_factor)

    tfr = mne.time_frequency.tfr_array_morlet(
        data[np.newaxis, np.newaxis, :],
        sfreq=original_sfreq,
        freqs=all_freqs,
        n_cycles=n_cycles,
        output="power",
        verbose=False,
    )
    power = tfr[0, 0, :, :]

    out: dict[str, list[float]] = {}
    offset = 0
    for band, freqs in zip(band_names, band_freqs):
        n = len(freqs)
        bp = power[offset : offset + n, :].mean(axis=0)
        offset += n
        decim = mne.filter.resample(
            bp, down=original_sfreq / target_sfreq, verbose=False
        )
        out[band] = (10.0 * np.log10(decim + 1e-20)).tolist()

    return out
