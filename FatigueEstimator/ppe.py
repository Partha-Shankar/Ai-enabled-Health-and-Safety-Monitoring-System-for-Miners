# ── models/ppe.py ─────────────────────────────────────────────────────────────
# YOLOPPEDetector: asynchronous PPE compliance detection using YOLOv8n.
# Runs in a background thread at ~10 inferences/second.  Person presence is
# overridden by the faster MediaPipe face-mesh path for instant feedback.
# Hysteresis thresholding prevents state flickering on borderline detections.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import time
import threading
from collections import deque
from typing import Dict

import cv2
import numpy as np

from config import (
    PPE_MODEL_PATH,
    PPE_CONFIDENCE_THRESHOLD,
    PPE_ON_RATIO,
    PPE_OFF_RATIO,
    PPE_INFERENCE_SIZE,
    PPE_INFERENCE_INTERVAL,
    PPE_HISTORY_WINDOW,
    PPE_ITEMS,
)

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ── Model search paths ────────────────────────────────────────────────────────
_MODEL_SEARCH_PATHS = [
    PPE_MODEL_PATH,
    'ppe_dataset/train/weights/best.pt',
    'ppe_dataset/weights/best.pt',
    'best.pt',
]

# ── COCO / custom label normalisation map ────────────────────────────────────
# Maps raw YOLO class names (from either COCO or a custom PPE dataset) to the
# canonical keys used throughout the system.
_LABEL_MAP: Dict[str, str] = {
    'person':  'person',  'worker':  'person',
    'helmet':  'helmet',  'hardhat': 'helmet',
    'goggle':  'goggles', 'glasses': 'goggles',
    'vest':    'vest',    'jacket':  'vest',
    'glove':   'gloves',
    'boot':    'boots',   'shoe':    'boots',
}


def _normalise_label(raw: str) -> str | None:
    """Return the canonical PPE key for a YOLO class name, or None."""
    raw = raw.lower()
    for fragment, canonical in _LABEL_MAP.items():
        if fragment in raw:
            return canonical
    return None


# ── Detector ──────────────────────────────────────────────────────────────────

class YOLOPPEDetector:
    """
    Background-threaded YOLO PPE detector with hysteresis state tracking.

    Usage
    -----
    detector = YOLOPPEDetector(PPE_MODEL_PATH)
    detector.submit(frame)          # non-blocking; call from capture loop
    state = detector.get_state()    # {'person': bool, 'helmet': bool, ...}
    detector.force_person(True)     # instant override from face-mesh
    """

    def __init__(self, model_path: str = PPE_MODEL_PATH) -> None:
        self.model:        object  = None
        self.is_available: bool    = False
        self._lock                 = threading.Lock()
        self._pending: np.ndarray | None = None
        self._stop                 = threading.Event()

        # Per-item detection history (hysteresis)
        self._history: Dict[str, deque] = {
            k: deque(maxlen=PPE_HISTORY_WINDOW) for k in PPE_ITEMS
        }
        self.current_state: Dict[str, bool] = {k: False for k in PPE_ITEMS}

        if _YOLO_AVAILABLE:
            self._load_model(model_path)
        else:
            print('⚠️  ultralytics not installed — PPE detection disabled')

        if self.is_available:
            t = threading.Thread(target=self._inference_loop, daemon=True)
            t.start()

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self, primary_path: str) -> None:
        search = [primary_path] + [p for p in _MODEL_SEARCH_PATHS if p != primary_path]
        for path in search:
            if os.path.exists(path):
                self.model = YOLO(path)
                self.model.overrides['verbose'] = False
                self.is_available = True
                print(f'✓ YOLO loaded: {path}')
                return
        print('⚠️  YOLO model not found — PPE detection disabled')

    # ── public API ────────────────────────────────────────────────────────────

    def submit(self, frame: np.ndarray) -> None:
        """
        Queue *frame* for the next inference cycle.
        Non-blocking: only the most recent frame is processed; stale frames
        are dropped automatically.
        """
        small = cv2.resize(frame, (PPE_INFERENCE_SIZE, PPE_INFERENCE_SIZE))
        with self._lock:
            self._pending = small

    def get_state(self) -> Dict[str, bool]:
        with self._lock:
            return self.current_state.copy()

    def force_person(self, present: bool) -> None:
        """
        Override person detection state from the faster MediaPipe path.
        Ensures person presence is flagged as soon as a face is detected,
        without waiting for the next YOLO inference cycle.
        """
        with self._lock:
            self._history['person'].append(1 if present else 0)
            h = self._history['person']
            if len(h) >= 2:
                ratio = sum(h) / len(h)
                if present and ratio >= 0.50:
                    self.current_state['person'] = True
                if not present and ratio <= 0.25:
                    self.current_state['person'] = False

    def reset(self) -> None:
        with self._lock:
            for k in PPE_ITEMS:
                self._history[k].clear()
                self.current_state[k] = False

    def draw(self, frame: np.ndarray, state: Dict[str, bool]) -> np.ndarray:
        """Render the PPE status overlay onto *frame* in-place."""
        if not self.is_available:
            return frame

        from utils.video_utils import draw_semi_transparent_rect, put_status_text
        draw_semi_transparent_rect(frame, 10, 10, 240, 160)
        cv2.putText(frame, 'PPE', (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        labels = [
            ('Person',  'person'),
            ('Helmet',  'helmet'),
            ('Goggles', 'goggles'),
            ('Vest',    'vest'),
            ('Gloves',  'gloves'),
            ('Boots',   'boots'),
        ]
        y = 58
        for label, key in labels:
            ok = state.get(key, False)
            put_status_text(frame, f"{'OK' if ok else 'XX'} {label}", (20, y), ok)
            y += 17

        return frame

    # ── inference loop ────────────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        """
        Background thread: dequeues frames, runs YOLO inference, and updates
        the hysteresis state for each PPE item.
        """
        while not self._stop.is_set():
            with self._lock:
                frame   = self._pending
                self._pending = None

            if frame is None:
                time.sleep(0.025)
                continue

            try:
                results = self.model(
                    frame,
                    conf=PPE_CONFIDENCE_THRESHOLD,
                    verbose=False,
                    device='cpu',
                    imgsz=PPE_INFERENCE_SIZE,
                )[0]

                seen: Dict[str, bool] = {k: False for k in PPE_ITEMS}

                if results.boxes is not None:
                    for box in results.boxes:
                        raw_name  = results.names[int(box.cls[0])]
                        canonical = _normalise_label(raw_name)
                        if canonical:
                            seen[canonical] = True

                with self._lock:
                    self._apply_hysteresis(seen)

            except Exception as exc:
                print(f'⚠️  PPE inference error: {exc}')

            time.sleep(PPE_INFERENCE_INTERVAL)

    def _apply_hysteresis(self, seen: Dict[str, bool]) -> None:
        """
        Update detection history and flip item state using hysteresis
        thresholds to prevent flickering.

        ON  threshold : detection ratio >= PPE_ON_RATIO
        OFF threshold : detection ratio <= PPE_OFF_RATIO
        """
        for k in PPE_ITEMS:
            self._history[k].append(1 if seen[k] else 0)
            h       = self._history[k]
            on_thr  = 0.50 if k == 'person' else PPE_ON_RATIO
            off_thr = 0.25 if k == 'person' else PPE_OFF_RATIO
            win     = 2    if k == 'person' else 5

            if len(h) >= win:
                ratio = sum(h) / len(h)
                if not self.current_state[k] and ratio >= on_thr:
                    self.current_state[k] = True
                elif self.current_state[k] and ratio <= off_thr:
                    self.current_state[k] = False
