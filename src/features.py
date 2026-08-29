"""Signal conditioning and the six scale-invariant descriptors used by the
feature-based detectors."""
import numpy as np
from scipy import signal as sps
from scipy.signal import butter, filtfilt

from config import BAND_HZ, BAND_ORDER, RAW_LENGTH


def bandpass(x, fs):
    """Zero-phase Butterworth band-pass over the ABR analysis band."""
    b, a = butter(BAND_ORDER, [BAND_HZ[0] / (fs / 2), BAND_HZ[1] / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def znorm(x):
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 1e-12 else x - np.nanmean(x)


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def window(x, fs, span_ms, onset=0):
    i = max(0, onset + int(round(span_ms[0] / 1000 * fs)))
    j = min(len(x), onset + int(round(span_ms[1] / 1000 * fs)))
    return x[i:j]


FEATURE_NAMES = [
    "rms_ratio", "p2p_ratio", "max_ratio", "peak_latency_ms",
    "band_energy_fraction", "spectral_centroid",
]


def engineered_features(x, fs, response_ms, noise_ms, onset=0):
    """Six scale-invariant descriptors of a single averaged run.

    Ratios are response-window over noise-window, so an arbitrary amplitude
    scaling of the recording cancels.
    """
    resp = window(x, fs, response_ms, onset)
    noise = window(x, fs, noise_ms, onset)
    eps = 1e-12
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else eps
    p2p = lambda v: float(v.max() - v.min()) if len(v) else eps
    amax = lambda v: float(np.max(np.abs(v))) if len(v) else eps

    freqs, power = sps.welch(resp, fs=fs, nperseg=min(256, len(resp)))
    in_band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])

    return np.array([
        rms(resp) / (rms(noise) + eps),
        p2p(resp) / (p2p(noise) + eps),
        amax(resp) / (amax(noise) + eps),
        response_ms[0] + float(np.argmax(np.abs(resp))) / fs * 1000.0,
        float(power[in_band].sum() / (power.sum() + eps)),
        float((freqs * power).sum() / (power.sum() + eps)),
    ])


def raw_vector(x, fs, span_ms, onset=0):
    """Recorded waveform resampled to a fixed length and z-normalised."""
    seg = window(x, fs, span_ms, onset)
    return znorm(sps.resample(seg, RAW_LENGTH))
