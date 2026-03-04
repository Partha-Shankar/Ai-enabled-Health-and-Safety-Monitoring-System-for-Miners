# ── models/vitals.py ──────────────────────────────────────────────────────────
# Remote vital sign estimators: heart rate (rPPG), SpO₂ (R-ratio), and
# respiration rate (optical-flow FFT).  Each estimator maintains its own
# time-stamped signal buffer and exposes a single .estimate() method.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from collections import deque
from typing import Optional, Tuple

import numpy as np

from config import (
    RPPG_WINDOW_SEC,
    HR_MIN_BPM, HR_MAX_BPM,
    SPO2_WINDOW_SEC,
    RESP_BUFFER_SEC,
    RESP_MIN_BPM, RESP_MAX_BPM,
    FOREHEAD_POINTS,
)
from utils.signal_utils import (
    bandpass_peak_freq,
    r_ratio_to_spo2,
    resample_uniform,
)


# ── Heart Rate (rPPG) ─────────────────────────────────────────────────────────

class HeartRateEstimator:
    """
    Remote photoplethysmography (rPPG) heart rate estimator.

    Extracts the mean green-channel intensity from a forehead ROI on each
    frame, buffers samples over a 15-second window, and applies FFT peak
    detection in the 40–180 BPM band to recover the cardiac frequency.

    The green channel is chosen because it has the highest signal-to-noise
    ratio for blood-volume pulse signals under typical ambient lighting.
    """

    def __init__(self, window_sec: float = RPPG_WINDOW_SEC) -> None:
        self._window_sec = window_sec
        self._buffer: deque = deque()       # (timestamp, green_mean)
        self._last_hr: Optional[float] = None

    # ── buffer management ─────────────────────────────────────────────────────

    def push(self, frame: np.ndarray, landmarks, frame_w: int, frame_h: int) -> None:
        """
        Extract the forehead ROI from *frame* and append the green-channel
        mean to the rolling buffer.  Called every N frames by VideoProcessor.
        """
        from utils.video_utils import forehead_roi_from_landmarks, clamp_roi
        try:
            x, y, w, h = forehead_roi_from_landmarks(
                landmarks, FOREHEAD_POINTS, frame_w, frame_h)
        except Exception:
            return

        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return

        now = time.time()
        self._buffer.append((now, float(np.mean(roi[:, :, 1]))))  # green channel
        self._evict()

    def _evict(self) -> None:
        now = time.time()
        while self._buffer and now - self._buffer[0][0] > self._window_sec:
            self._buffer.popleft()

    # ── estimation ────────────────────────────────────────────────────────────

    def estimate(self) -> Optional[float]:
        """
        Return the current HR estimate in BPM, or None if insufficient data.
        Requires at least 8 seconds of buffered samples.
        """
        if len(self._buffer) < 8:
            return None

        times = np.array([t for t, _ in self._buffer])
        vals  = np.array([v for _, v in self._buffer])
        duration = times[-1] - times[0]
        if duration < 8.0:
            return None

        _, uniform_vals, fs = resample_uniform(times, vals)
        freq = bandpass_peak_freq(
            uniform_vals, fs,
            HR_MIN_BPM / 60.0,
            HR_MAX_BPM / 60.0,
        )
        if freq is None:
            return None

        hr = freq * 60.0
        self._last_hr = hr
        return hr

    @property
    def last(self) -> Optional[float]:
        return self._last_hr

    def reset(self) -> None:
        self._buffer.clear()
        self._last_hr = None


# ── SpO₂ ──────────────────────────────────────────────────────────────────────

class SpO2Estimator:
    """
    Camera-based SpO₂ estimator using the AC/DC R-ratio method.

    Simultaneously buffers red and green channel forehead ROI means.
    SpO₂ is derived from the ratio of their normalised AC amplitudes:

        R    = (std(red) / mean(red)) / (std(green) / mean(green))
        SpO₂ = 110 − 25 × R   (empirical Beer–Lambert approximation)

    Accuracy degrades under poor lighting or significant motion.
    """

    def __init__(self, window_sec: float = SPO2_WINDOW_SEC) -> None:
        self._window_sec = window_sec
        self._red_buf:   deque = deque()    # (timestamp, red_mean)
        self._green_buf: deque = deque()    # (timestamp, green_mean)
        self._last_spo2: Optional[float] = None

    def push(self, frame: np.ndarray, landmarks, frame_w: int, frame_h: int) -> None:
        from utils.video_utils import forehead_roi_from_landmarks
        try:
            x, y, w, h = forehead_roi_from_landmarks(
                landmarks, FOREHEAD_POINTS, frame_w, frame_h)
        except Exception:
            return

        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return

        now = time.time()
        self._green_buf.append((now, float(np.mean(roi[:, :, 1]))))
        self._red_buf.append((now,   float(np.mean(roi[:, :, 2]))))
        self._evict()

    def _evict(self) -> None:
        now = time.time()
        for buf in (self._red_buf, self._green_buf):
            while buf and now - buf[0][0] > self._window_sec:
                buf.popleft()

    def estimate(self) -> Optional[float]:
        if len(self._red_buf) < 8 or len(self._green_buf) < 8:
            return None

        # align on overlapping time range
        t0 = max(self._red_buf[0][0],   self._green_buf[0][0])
        t1 = min(self._red_buf[-1][0],  self._green_buf[-1][0])
        if t1 - t0 < 6.0:
            return None

        def _extract(buf):
            ts = np.array([t for t, _ in buf])
            vs = np.array([v for _, v in buf])
            mask = (ts >= t0) & (ts <= t1)
            return vs[mask]

        rv = _extract(self._red_buf)
        gv = _extract(self._green_buf)

        spo2 = r_ratio_to_spo2(rv, gv)
        if spo2 is not None:
            self._last_spo2 = spo2
        return spo2

    @property
    def last(self) -> Optional[float]:
        return self._last_spo2

    def reset(self) -> None:
        self._red_buf.clear()
        self._green_buf.clear()
        self._last_spo2 = None


# ── Respiration Rate ──────────────────────────────────────────────────────────

class RespirationEstimator:
    """
    Respiration rate estimator using dense optical flow on a chest ROI.

    The mean vertical (y-axis) flow magnitude in the chest region is used
    as a proxy for thoracic expansion and contraction.  Samples are buffered
    over a 30-second window and transformed via FFT to extract the dominant
    frequency in the 6–30 BPM physiological band.
    """

    def __init__(self, window_sec: float = RESP_BUFFER_SEC) -> None:
        self._window_sec = window_sec
        self._buffer: deque = deque()       # (timestamp, flow_magnitude)
        self._last_bpm: Optional[float] = None

    def push(self, flow_magnitude: float) -> None:
        """Append a single optical-flow sample with the current timestamp."""
        now = time.time()
        self._buffer.append((now, float(flow_magnitude)))
        while self._buffer and now - self._buffer[0][0] > self._window_sec:
            self._buffer.popleft()

    def estimate(self) -> Optional[float]:
        """
        Return the current respiration rate in BPM, or None.
        Requires at least 6 seconds of buffered data.
        """
        if len(self._buffer) < 8:
            return None

        times = np.array([t for t, _ in self._buffer])
        vals  = np.array([v for _, v in self._buffer])
        duration = times[-1] - times[0]
        if duration < 6.0:
            return None

        _, uniform_vals, fs = resample_uniform(times, vals)
        freq = bandpass_peak_freq(
            uniform_vals, fs,
            RESP_MIN_BPM / 60.0,
            RESP_MAX_BPM / 60.0,
        )
        if freq is None:
            return None

        bpm = freq * 60.0
        self._last_bpm = bpm
        return bpm

    @property
    def last(self) -> Optional[float]:
        return self._last_bpm

    def reset(self) -> None:
        self._buffer.clear()
        self._last_bpm = None
