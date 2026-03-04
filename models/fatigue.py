# ── models/fatigue.py ─────────────────────────────────────────────────────────
# FatigueEstimator: composite fatigue scoring from eye, yawn, and respiration
# signals.  BlinkDetector: adaptive-baseline blink counting using a rolling
# 80th-percentile open-eye EAR baseline.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import (
    EAR_THRESHOLD,
    CONSEC_FRAMES_MICROSLEEP,
    PERCLOS_WINDOW_SEC,
    BLINK_EAR_DROP_RATIO,
    BLINK_MIN_FRAMES,
    BLINK_MAX_FRAMES,
    BLINK_MIN_SEPARATION_MS,
    BLINK_BASELINE_PERCENTILE,
    BLINK_BASELINE_MIN_SAMPLES,
    FATIGUE_EMA_ALPHA,
    WEIGHT_EYE,
    WEIGHT_YAWN,
    WEIGHT_RESP,
    YAWN_SATURATION_COUNT,
    MICROSLEEP_DECAY,
)


# ── Blink Detector ────────────────────────────────────────────────────────────

class BlinkDetector:
    """
    Adaptive-baseline blink detector.

    The open-eye EAR baseline is tracked as a rolling 80th-percentile over
    recent *open* samples (EAR > current threshold).  A blink is confirmed
    when the eye stays closed for 2–12 frames and then re-opens, with at
    least BLINK_MIN_SEPARATION_MS since the previous confirmed blink.

    Consecutive fast blinks are also counted provided each closure satisfies
    the duration constraint.
    """

    def __init__(self) -> None:
        self.count          = 0
        self.closed_frames  = 0
        self._in_blink      = False
        self._last_blink_ts = 0.0

        self._baseline_buf  = deque(maxlen=150)
        self.baseline       = 0.28
        self.threshold      = 0.20

    # ── private ───────────────────────────────────────────────────────────────

    def _update_baseline(self, ear_avg: float) -> None:
        if ear_avg > self.threshold:
            self._baseline_buf.append(ear_avg)
        if len(self._baseline_buf) >= BLINK_BASELINE_MIN_SAMPLES:
            self.baseline  = float(np.percentile(
                list(self._baseline_buf), BLINK_BASELINE_PERCENTILE))
            self.threshold = max(0.15, self.baseline * BLINK_EAR_DROP_RATIO)

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, ear_l: float, ear_r: float, ts: float) -> bool:
        """
        Feed one frame of left/right EAR values.

        Parameters
        ----------
        ear_l, ear_r : per-eye aspect ratios
        ts           : current timestamp (seconds)

        Returns True if a blink was confirmed on this frame transition.
        """
        ear = (ear_l + ear_r) / 2.0
        self._update_baseline(ear)

        is_closed      = ear < self.threshold
        blink_detected = False

        if is_closed:
            self.closed_frames += 1
            self._in_blink = True
        else:
            if self._in_blink:
                gap_ms = (ts - self._last_blink_ts) * 1000.0
                dur_ok = BLINK_MIN_FRAMES <= self.closed_frames <= BLINK_MAX_FRAMES

                if dur_ok:
                    self.count         += 1
                    self._last_blink_ts = ts
                    blink_detected      = True

                self.closed_frames = 0
                self._in_blink     = False

        return blink_detected

    def get_count(self) -> int:
        return self.count

    def reset(self) -> None:
        self.__init__()


# ── PERCLOS Tracker ───────────────────────────────────────────────────────────

class PerclosTracker:
    """
    Rolling-window PERCLOS (Percentage of Eye Closure) calculator.

    Maintains a deque of boolean closed-eye flags over a configurable
    window and returns the fraction of frames where EAR < threshold.
    """

    def __init__(self, window_sec: float = PERCLOS_WINDOW_SEC,
                 fps: int = 30) -> None:
        maxlen = int(window_sec * fps)
        self._window: deque = deque(maxlen=maxlen)

    def update(self, ear: Optional[float], threshold: float = EAR_THRESHOLD) -> None:
        self._window.append(ear is not None and ear < threshold)

    def value(self) -> float:
        if not self._window:
            return 0.0
        return float(sum(self._window) / len(self._window))

    def reset(self) -> None:
        self._window.clear()


# ── Fatigue Estimator ─────────────────────────────────────────────────────────

@dataclass
class FatigueEstimator:
    """
    Composite fatigue score (0–100) from three weighted sub-signals.

    Sub-scores
    ----------
    eye_score  = 0.7 × PERCLOS + 0.3 × (1 − e^(−λ × microsleep_count))
    yawn_score = min(yawn_count / YAWN_SATURATION_COUNT, 1.0)
    resp_score = deviation from 12–20 BPM normal range, clipped to [0, 1]

    Final (raw) score = WEIGHT_EYE × eye + WEIGHT_YAWN × yawn + WEIGHT_RESP × resp
    Smoothed via EMA with FATIGUE_EMA_ALPHA to avoid noisy output.
    """

    weight_eye:  float = WEIGHT_EYE
    weight_yawn: float = WEIGHT_YAWN
    weight_resp: float = WEIGHT_RESP
    smoothed:    float = 0.0

    def _eye_score(self, perclos: float, microsleep_count: int) -> float:
        ms_factor = 1.0 - math.exp(-MICROSLEEP_DECAY * microsleep_count)
        return float(np.clip(0.7 * perclos + 0.3 * ms_factor, 0.0, 1.0))

    def _yawn_score(self, yawn_count: int) -> float:
        return float(np.clip(yawn_count / YAWN_SATURATION_COUNT, 0.0, 1.0))

    def _resp_score(self, resp_bpm: float) -> float:
        if resp_bpm <= 0:
            return 0.0
        if resp_bpm < 12:
            return float(np.clip((12.0 - resp_bpm) / 12.0, 0.0, 1.0))
        if resp_bpm > 20:
            return float(np.clip((resp_bpm - 20.0) / 20.0, 0.0, 1.0))
        return 0.0

    def compute_raw(
        self,
        perclos:         float,
        yawn_count:      int,
        microsleep_count:int,
        resp_bpm:        float,
    ) -> float:
        """Return un-smoothed fatigue score in [0, 100]."""
        eye  = self._eye_score(perclos, microsleep_count)
        yawn = self._yawn_score(yawn_count)
        resp = self._resp_score(resp_bpm)
        raw  = self.weight_eye * eye + self.weight_yawn * yawn + self.weight_resp * resp
        return float(np.clip(raw * 100.0, 0.0, 100.0))

    def smooth(self, raw: float) -> float:
        """Apply EMA smoothing and return the updated smoothed score."""
        self.smoothed = FATIGUE_EMA_ALPHA * raw + (1.0 - FATIGUE_EMA_ALPHA) * self.smoothed
        return self.smoothed

    def score(
        self,
        perclos:         float,
        yawn_count:      int,
        microsleep_count:int,
        resp_bpm:        float,
    ) -> float:
        """Convenience: compute raw then return smoothed score."""
        return self.smooth(self.compute_raw(perclos, yawn_count, microsleep_count, resp_bpm))

    def reset(self) -> None:
        self.smoothed = 0.0
