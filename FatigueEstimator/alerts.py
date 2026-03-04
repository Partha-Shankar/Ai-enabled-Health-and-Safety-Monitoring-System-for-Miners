# ── models/alerts.py ──────────────────────────────────────────────────────────
# AlertManager: thread-safe alert generation with per-key deduplication,
# configurable cooldown windows, and a bounded history deque.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

from config import (
    ALERT_COOLDOWN_SEC,
    ALERT_HISTORY_MAXLEN,
    ALERT_PERCLOS,
    ALERT_FATIGUE,
    ALERT_HR_LOW,
    ALERT_HR_HIGH,
    ALERT_SPO2_LOW,
    ALERT_RESP_LOW,
    ALERT_RESP_HIGH,
    ALERT_MICROSLEEP_COUNT,
    ALERT_YAWN_COUNT,
    PPE_ALERT_ENABLED,
    PPE_ITEMS,
)


# ── Alert record ──────────────────────────────────────────────────────────────

class Alert:
    """Immutable record for a single triggered alert."""

    __slots__ = ('timestamp', 'name', 'message', 'severity')

    SEVERITY_INFO    = 'info'
    SEVERITY_WARNING = 'warning'
    SEVERITY_CRITICAL= 'critical'

    def __init__(self, name: str, message: str,
                 severity: str = SEVERITY_WARNING) -> None:
        self.timestamp = datetime.now().strftime('%H:%M:%S')
        self.name      = name
        self.message   = message
        self.severity  = severity

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'name':      self.name,
            'message':   self.message,
            'severity':  self.severity,
        }


# ── Alert Manager ─────────────────────────────────────────────────────────────

class AlertManager:
    """
    Thread-safe alert manager with cooldown-based deduplication.

    Each unique (name, message) pair is rate-limited to once per
    ALERT_COOLDOWN_SEC seconds.  The last ALERT_HISTORY_MAXLEN alerts
    are kept in a bounded deque for frontend polling.

    Usage
    -----
    am = AlertManager()
    am.trigger('HIGH_FATIGUE', 'Score=82.3', Alert.SEVERITY_WARNING)
    alerts = am.get_alerts()        # list of dicts
    am.clear()
    """

    def __init__(
        self,
        cooldown_sec: float = ALERT_COOLDOWN_SEC,
        maxlen:       int   = ALERT_HISTORY_MAXLEN,
    ) -> None:
        self._history: deque  = deque(maxlen=maxlen)
        self._lock            = threading.Lock()
        self._last_ts: Dict[str, float] = {}
        self._cooldown        = cooldown_sec

    # ── core ──────────────────────────────────────────────────────────────────

    def trigger(
        self,
        name:     str,
        message:  str,
        severity: str = Alert.SEVERITY_WARNING,
    ) -> bool:
        """
        Emit an alert if the cooldown for this (name, message) pair has elapsed.

        Returns True if the alert was recorded, False if suppressed.
        """
        key = f'{name}_{message}'
        now = time.time()

        with self._lock:
            if now - self._last_ts.get(key, 0.0) < self._cooldown:
                return False
            alert = Alert(name, message, severity)
            self._history.append(alert)
            self._last_ts[key] = now
            return True

    def get_alerts(self) -> List[dict]:
        with self._lock:
            return [a.to_dict() for a in self._history]

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._last_ts.clear()

    # backward-compat alias used in app.py
    def clear_alerts(self) -> None:
        self.clear()

    # ── evaluation helpers ────────────────────────────────────────────────────

    def evaluate_vitals(
        self,
        perclos:          float,
        fatigue_score:    float,
        microsleep_count: int,
        yawn_count:       int,
        resp_bpm:         float,
        hr_bpm:           float,
        spo2:             float,
    ) -> None:
        """
        Check all vital-sign thresholds and trigger alerts as needed.
        Designed to be called once per processed frame.
        """
        if perclos >= ALERT_PERCLOS:
            self.trigger('HIGH_PERCLOS',
                         f'PERCLOS={perclos:.2f}',
                         Alert.SEVERITY_WARNING)

        if fatigue_score >= ALERT_FATIGUE:
            self.trigger('HIGH_FATIGUE',
                         f'Score={fatigue_score:.1f}',
                         Alert.SEVERITY_CRITICAL)

        if microsleep_count >= ALERT_MICROSLEEP_COUNT:
            self.trigger('MICROSLEEP',
                         f'Count={microsleep_count}',
                         Alert.SEVERITY_CRITICAL)

        if yawn_count >= ALERT_YAWN_COUNT:
            self.trigger('FREQUENT_YAWNS',
                         f'Yawns={yawn_count}',
                         Alert.SEVERITY_WARNING)

        if resp_bpm > 0:
            if resp_bpm < ALERT_RESP_LOW:
                self.trigger('RESP_LOW',  f'{resp_bpm:.1f} bpm', Alert.SEVERITY_WARNING)
            elif resp_bpm > ALERT_RESP_HIGH:
                self.trigger('RESP_HIGH', f'{resp_bpm:.1f} bpm', Alert.SEVERITY_WARNING)

        if hr_bpm > 0:
            if hr_bpm < ALERT_HR_LOW:
                self.trigger('HR_LOW',  f'{hr_bpm:.0f} bpm', Alert.SEVERITY_CRITICAL)
            elif hr_bpm > ALERT_HR_HIGH:
                self.trigger('HR_HIGH', f'{hr_bpm:.0f} bpm', Alert.SEVERITY_CRITICAL)

        if spo2 > 0 and spo2 < ALERT_SPO2_LOW:
            self.trigger('LOW_SPO2', f'{spo2:.1f}%', Alert.SEVERITY_CRITICAL)

    def evaluate_ppe(self, ppe_state: dict) -> None:
        """
        Trigger PPE non-compliance alerts when a person is confirmed present
        but one or more equipment items are not detected.
        """
        if not PPE_ALERT_ENABLED:
            return
        if not ppe_state.get('person'):
            return

        items = [
            ('helmet',  'NO_HELMET',  'Helmet not detected'),
            ('goggles', 'NO_GOGGLES', 'Goggles not detected'),
            ('vest',    'NO_VEST',    'Vest not detected'),
            ('gloves',  'NO_GLOVES',  'Gloves not detected'),
            ('boots',   'NO_BOOTS',   'Boots not detected'),
        ]
        for key, alert_name, message in items:
            if not ppe_state.get(key, False):
                self.trigger(alert_name, message, Alert.SEVERITY_WARNING)
