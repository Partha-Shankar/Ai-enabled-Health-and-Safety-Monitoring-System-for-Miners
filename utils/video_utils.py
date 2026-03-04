# ── utils/video_utils.py ──────────────────────────────────────────────────────
# Frame-level OpenCV helpers: ROI extraction, scaling, annotation, and stream
# encoding utilities used across VideoProcessor and YOLOPPEDetector.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

from config import (
    FLOW_ROI_REL,
    PPE_INFERENCE_SIZE,
    STREAM_JPEG_QUALITY,
)


# ── Frame scaling ─────────────────────────────────────────────────────────────

def scale_frame(frame: np.ndarray, factor: float) -> np.ndarray:
    """
    Resize *frame* by a scalar *factor* using bilinear interpolation.
    factor < 1 → downscale; factor > 1 → upscale.
    """
    h, w  = frame.shape[:2]
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def scale_frame_to(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to an absolute (width, height) in pixels."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def yolo_input_frame(frame: np.ndarray) -> np.ndarray:
    """Downscale to PPE_INFERENCE_SIZE × PPE_INFERENCE_SIZE for YOLO inference."""
    return cv2.resize(frame, (PPE_INFERENCE_SIZE, PPE_INFERENCE_SIZE),
                      interpolation=cv2.INTER_LINEAR)


# ── ROI helpers ───────────────────────────────────────────────────────────────

def clamp_roi(
    x: int, y: int, rw: int, rh: int,
    frame_w: int, frame_h: int,
    min_dim: int = 8,
) -> Tuple[int, int, int, int]:
    """
    Clamp an ROI rectangle so it stays inside the frame and has minimum size.

    Returns
    -------
    (x, y, rw, rh) — safe, clamped values
    """
    x  = max(0, min(x,  frame_w - 1))
    y  = max(0, min(y,  frame_h - 1))
    rw = max(min_dim, min(rw, frame_w - x))
    rh = max(min_dim, min(rh, frame_h - y))
    return x, y, rw, rh


def relative_roi(frame: np.ndarray, rel: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Extract a sub-region using relative (0–1) coordinates.

    Parameters
    ----------
    frame : BGR or grayscale image
    rel   : (x_frac, y_frac, w_frac, h_frac)
    """
    h, w   = frame.shape[:2]
    rx, ry, rw, rh = rel
    x, y   = int(w * rx), int(h * ry)
    ew, eh = int(w * rw), int(h * rh)
    x, y, ew, eh = clamp_roi(x, y, ew, eh, w, h)
    return frame[y:y + eh, x:x + ew]


def default_chest_roi(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Fallback chest ROI using the fixed relative coordinates from config.
    Used when MediaPipe face landmarks are unavailable.
    """
    h, w = frame.shape[:2]
    rx, ry, rw, rh = FLOW_ROI_REL
    return int(w * rx), int(h * ry), int(w * rw), int(h * rh)


def forehead_roi_from_landmarks(
    landmarks,
    forehead_indices: list,
    frame_w: int,
    frame_h: int,
    x_frac: float = 0.18,
    y_frac: float = 0.08,
) -> Tuple[int, int, int, int]:
    """
    Derive the forehead ROI bounding box from MediaPipe landmark positions.

    The ROI is centred on the mean (x, y) of *forehead_indices* landmarks and
    sized as a fraction of the full frame dimensions.

    Returns (x, y, w, h) in absolute pixels, clamped to frame bounds.
    """
    xs = np.array([landmarks[p].x for p in forehead_indices]) * frame_w
    ys = np.array([landmarks[p].y for p in forehead_indices]) * frame_h
    cx, cy = int(np.mean(xs)), int(np.mean(ys))

    ww = int(frame_w * x_frac)
    hh = int(frame_h * y_frac)

    x = max(0, cx - ww // 2)
    y = max(0, int(cy - hh * 0.6))
    return clamp_roi(x, y, ww, hh, frame_w, frame_h)


def chest_roi_from_landmarks(
    landmarks,
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int, int, int]:
    """
    Estimate chest ROI from MediaPipe face mesh by projecting below the face
    bounding box.  Falls back to *default_chest_roi* dimensions on failure.
    """
    ys = [lm.y for lm in landmarks]
    xs = [lm.x for lm in landmarks]

    max_y = max(ys) * frame_h
    min_x = min(xs) * frame_w
    max_x = max(xs) * frame_w

    fw  = max_x - min_x
    rw  = int(max(fw * 1.0, frame_w * 0.25))
    rx  = int(max(0, (min_x + max_x) / 2 - rw / 2))
    ry  = int(min(frame_h - 1, max_y + 10))
    rh  = int(min(frame_h - ry - 10, frame_h * 0.22))

    if rh <= 10 or rw <= 10:
        rx, ry, rw, rh = FLOW_ROI_REL
        rx  = int(frame_w * rx)
        ry  = int(frame_h * ry)
        rw  = int(frame_w * rw)
        rh  = int(frame_h * rh)

    return clamp_roi(rx, ry, rw, rh, frame_w, frame_h)


# ── Optical flow ──────────────────────────────────────────────────────────────

def farneback_mean_vertical(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    pyr_scale: float = 0.5,
    levels:    int   = 2,
    winsize:   int   = 10,
    iterations:int   = 2,
    poly_n:    int   = 5,
    poly_sigma:float = 1.1,
) -> float:
    """
    Compute dense optical flow (Farneback) and return the mean vertical
    displacement of the ROI — used as a respiration motion proxy.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=pyr_scale, levels=levels, winsize=winsize,
        iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma,
        flags=0,
    )
    return float(np.mean(flow[..., 1]))   # y-component → vertical motion


# ── JPEG encoding ─────────────────────────────────────────────────────────────

_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]


def encode_jpeg(frame: np.ndarray) -> Optional[bytes]:
    """
    Encode *frame* to JPEG bytes at the configured stream quality.
    Returns None if encoding fails.
    """
    ok, buf = cv2.imencode('.jpg', frame, _ENCODE_PARAMS)
    return buf.tobytes() if ok else None


def mjpeg_frame(data: bytes) -> bytes:
    """
    Wrap raw JPEG *data* in a multipart/x-mixed-replace boundary chunk.
    Suitable for direct yielding in a Flask streaming response.
    """
    return (
        b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n'
        b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n'
        + data + b'\r\n'
    )


# ── Overlay drawing ───────────────────────────────────────────────────────────

def draw_semi_transparent_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int] = (20, 20, 40),
    alpha: float = 0.65,
) -> np.ndarray:
    """
    Blend a filled rectangle onto *frame* with opacity *alpha*.
    Returns the modified frame in-place.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return frame


def put_status_text(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    ok: bool,
    font_scale: float = 0.45,
    thickness: int    = 1,
) -> None:
    """
    Draw *text* at *pos* in green if *ok*, red otherwise.
    Used for PPE item status labels on the video overlay.
    """
    color = (0, 220, 80) if ok else (0, 60, 220)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)
