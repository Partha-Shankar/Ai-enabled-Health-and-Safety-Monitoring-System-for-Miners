# ── utils/signal_utils.py ─────────────────────────────────────────────────────
# Low-level signal processing utilities used by the vitals and fatigue modules.
# All functions are stateless and operate on raw numpy arrays or landmark lists.
# ──────────────────────────────────────────────────────────────────────────────

import math
import numpy as np
from scipy.fft import rfft, rfftfreq
from typing import Optional, List, Tuple

from config import (
    LEFT_EYE_IDX, RIGHT_EYE_IDX,
    UPPER_LIP_IDX, LOWER_LIP_IDX,
    MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX,
    SPO2_R_SCALE, SPO2_R_OFFSET,
    SPO2_CLIP_LOW, SPO2_CLIP_HIGH,
)


# ── Geometry ──────────────────────────────────────────────────────────────────

def safe_div(a: float, b: float) -> float:
    """Division that returns 0.0 instead of raising ZeroDivisionError."""
    return a / b if b != 0 else 0.0


def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ── Eye / Mouth aspect ratios ─────────────────────────────────────────────────

def eye_aspect_ratio(landmarks, idx_list: List[int], w: int, h: int) -> float:
    """
    Compute EAR (Eye Aspect Ratio) from 6 MediaPipe face-mesh landmarks.

    Formula (Soukupová & Čech, 2016):
        EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)

    Parameters
    ----------
    landmarks : MediaPipe NormalizedLandmarkList
    idx_list  : 6 landmark indices [p1, p2, p3, p4, p5, p6]
    w, h      : frame width and height for de-normalisation
    """
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idx_list]
    p1, p2, p3, p4, p5, p6 = pts
    vertical   = euclid(p2, p6) + euclid(p3, p5)
    horizontal = 2.0 * euclid(p1, p4)
    return safe_div(vertical, horizontal)


def mouth_aspect_ratio(landmarks, w: int, h: int) -> float:
    """
    Compute MAR (Mouth Aspect Ratio) for yawn detection.

    MAR = ||upper_lip − lower_lip|| / ||mouth_left − mouth_right||
    """
    up  = (landmarks[UPPER_LIP_IDX].x * w,   landmarks[UPPER_LIP_IDX].y * h)
    low = (landmarks[LOWER_LIP_IDX].x * w,   landmarks[LOWER_LIP_IDX].y * h)
    lft = (landmarks[MOUTH_LEFT_IDX].x * w,  landmarks[MOUTH_LEFT_IDX].y * h)
    rgt = (landmarks[MOUTH_RIGHT_IDX].x * w, landmarks[MOUTH_RIGHT_IDX].y * h)
    return safe_div(euclid(up, low), euclid(lft, rgt))


def mean_ear(landmarks, w: int, h: int) -> float:
    """Average EAR across both eyes."""
    l = eye_aspect_ratio(landmarks, LEFT_EYE_IDX,  w, h)
    r = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
    return (l + r) / 2.0


# ── Frequency-domain peak detection ──────────────────────────────────────────

def bandpass_peak_freq(
    signal: np.ndarray,
    fs: float,
    lo_hz: float,
    hi_hz: float,
) -> Optional[float]:
    """
    Return the dominant frequency (Hz) of *signal* within [lo_hz, hi_hz].

    Steps:
      1. Mean-centre the signal.
      2. Apply real-valued FFT.
      3. Zero out bins outside the band of interest.
      4. Return the frequency of the highest-magnitude bin.

    Returns None if the signal is too short or no bin falls in the band.
    """
    if len(signal) < 4:
        return None

    x   = signal - np.mean(signal)
    yf  = np.abs(rfft(x))
    xf  = rfftfreq(len(x), 1.0 / fs)

    mask = (xf >= lo_hz) & (xf <= hi_hz)
    if not np.any(mask):
        return None

    yf_masked = np.where(mask, yf, 0.0)
    freq      = float(xf[int(np.argmax(yf_masked))])
    return freq if freq > 0 else None


# ── SpO₂ from R-ratio ─────────────────────────────────────────────────────────

def r_ratio_to_spo2(red_signal: np.ndarray, green_signal: np.ndarray) -> Optional[float]:
    """
    Estimate SpO₂ from rPPG red and green channel signals.

    Uses the empirical Beer–Lambert approximation:
        R   = (AC_red / DC_red) / (AC_green / DC_green)
        SpO₂ = 110 − 25 × R

    Returns None if either buffer is too short or DC values are near zero.
    """
    if len(red_signal) < 4 or len(green_signal) < 4:
        return None

    dc_r = np.mean(red_signal)
    dc_g = np.mean(green_signal)
    if dc_r < 1e-6 or dc_g < 1e-6:
        return None

    ac_r = np.std(red_signal)
    ac_g = np.std(green_signal)

    R    = (ac_r / dc_r) / (ac_g / dc_g + 1e-12)
    spo2 = float(np.clip(SPO2_R_OFFSET - SPO2_R_SCALE * R, SPO2_CLIP_LOW, SPO2_CLIP_HIGH))
    return spo2


# ── Signal resampling ─────────────────────────────────────────────────────────

def resample_uniform(
    timestamps: np.ndarray,
    values: np.ndarray,
    n_out: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Linearly interpolate an irregularly-sampled signal onto a uniform grid.

    Parameters
    ----------
    timestamps : 1-D array of sample times (seconds)
    values     : 1-D array of sample values, same length
    n_out      : number of output samples; defaults to len(values)

    Returns
    -------
    uniform_times  : evenly spaced time array
    uniform_values : interpolated values
    fs             : implied sample rate (Hz)
    """
    if n_out is None:
        n_out = len(values)
    t0, t1        = timestamps[0], timestamps[-1]
    duration      = max(t1 - t0, 1e-6)
    fs            = n_out / duration
    uniform_times = np.linspace(t0, t1, n_out)
    uniform_vals  = np.interp(uniform_times, timestamps, values - np.mean(values))
    return uniform_times, uniform_vals, fs


# ── Exponential moving average ────────────────────────────────────────────────

def ema_update(current: Optional[float], new_value: float, alpha: float) -> float:
    """
    Single-step EMA update.
    Returns new_value directly on the first call (current is None).
    """
    if current is None:
        return new_value
    return alpha * new_value + (1.0 - alpha) * current


# ── Rolling percentile baseline ───────────────────────────────────────────────

def rolling_percentile(buffer: list, pct: float) -> float:
    """
    Compute *pct*-th percentile over a list of floats.
    Returns 0.0 for empty buffers.
    """
    if not buffer:
        return 0.0
    return float(np.percentile(buffer, pct))
