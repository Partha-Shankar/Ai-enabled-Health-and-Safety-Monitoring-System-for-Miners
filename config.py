# ── config.py ─────────────────────────────────────────────────────────────────
# Central configuration for the AI-Enabled Health & Safety Monitoring System.
# All runtime thresholds, model paths, and tunable parameters live here.
# Import this module wherever constants are needed instead of hardcoding values.
# ──────────────────────────────────────────────────────────────────────────────

# ── Eye / Blink ───────────────────────────────────────────────────────────────
EAR_THRESHOLD            = 0.21          # below this → eye considered closed
CONSEC_FRAMES_MICROSLEEP = 18            # ~0.6 s at 30 fps
EAR_EMA_ALPHA            = 0.60          # exponential moving average smoothing

BLINK_EAR_DROP_RATIO     = 0.72          # closed when EAR < baseline × this
BLINK_MIN_FRAMES         = 2
BLINK_MAX_FRAMES         = 12
BLINK_MIN_SEPARATION_MS  = 120           # prevents double-counting rapid blinks
BLINK_BASELINE_PERCENTILE= 80            # rolling percentile for open-eye baseline
BLINK_BASELINE_MIN_SAMPLES = 30

# ── Yawn / MAR ────────────────────────────────────────────────────────────────
MAR_YAWN_THRESHOLD       = 0.50
MAR_SUSTAIN_TIME         = 0.8           # seconds mouth must stay open
MAR_SLOPE_WINDOW         = 0.8           # seconds for slope computation
MAR_SLOPE_MAX            = 0.9           # reject sudden noise spikes
YAWN_MIN_SEPARATION      = 2.5           # seconds between counted yawns

# ── PERCLOS ───────────────────────────────────────────────────────────────────
PERCLOS_WINDOW_SEC       = 60.0          # rolling window length in seconds

# ── Respiration ───────────────────────────────────────────────────────────────
RESP_BUFFER_SEC          = 30.0
RESP_MIN_BPM             = 6
RESP_MAX_BPM             = 30
FLOW_ROI_REL             = (0.35, 0.6, 0.30, 0.25)   # (x, y, w, h) as fractions

# ── rPPG / Heart Rate ─────────────────────────────────────────────────────────
RPPG_WINDOW_SEC          = 15.0
HR_MIN_BPM               = 40
HR_MAX_BPM               = 180

# ── SpO₂ ──────────────────────────────────────────────────────────────────────
SPO2_WINDOW_SEC          = 15.0
SPO2_R_SCALE             = 25.0          # coefficient in 110 − 25R formula
SPO2_R_OFFSET            = 110.0
SPO2_CLIP_LOW            = 50.0
SPO2_CLIP_HIGH           = 100.0

# ── MediaPipe Face Mesh landmark indices ──────────────────────────────────────
LEFT_EYE_IDX   = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
UPPER_LIP_IDX  = 13
LOWER_LIP_IDX  = 14
MOUTH_LEFT_IDX = 78
MOUTH_RIGHT_IDX= 308
FOREHEAD_POINTS= [10, 338, 297, 332]

FACE_MESH_SCALE          = 0.75          # downscale factor before MediaPipe inference
FACE_MESH_MAX_FACES      = 1
FACE_MESH_MIN_DETECT_CONF= 0.5
FACE_MESH_MIN_TRACK_CONF = 0.5

# ── PPE / YOLO ────────────────────────────────────────────────────────────────
PPE_MODEL_PATH           = 'yolov8n.pt'
PPE_CONFIDENCE_THRESHOLD = 0.40
PPE_ALERT_ENABLED        = True
PPE_ON_RATIO             = 0.40          # detection ratio to flip state ON
PPE_OFF_RATIO            = 0.25          # detection ratio to flip state OFF
PPE_INFERENCE_SIZE       = 320           # downscaled resolution for YOLO
PPE_INFERENCE_INTERVAL   = 0.10         # seconds between YOLO runs (~10 fps)
PPE_HISTORY_WINDOW       = 8            # frames kept in hysteresis buffer

PPE_ITEMS = ('person', 'helmet', 'goggles', 'vest', 'gloves', 'boots')

# ── Fatigue Estimator weights ─────────────────────────────────────────────────
FATIGUE_EMA_ALPHA        = 0.12
WEIGHT_EYE               = 0.45
WEIGHT_YAWN              = 0.30
WEIGHT_RESP              = 0.25
YAWN_SATURATION_COUNT    = 5             # yawn_score saturates at this count
MICROSLEEP_DECAY         = 0.5           # λ in 1 − e^(−λ × ms_count)

# ── Alert thresholds ──────────────────────────────────────────────────────────
ALERT_PERCLOS            = 0.35
ALERT_FATIGUE            = 75.0
ALERT_HR_LOW             = 45.0
ALERT_HR_HIGH            = 120.0
ALERT_SPO2_LOW           = 94.0
ALERT_RESP_LOW           = 8.0
ALERT_RESP_HIGH          = 25.0
ALERT_MICROSLEEP_COUNT   = 1
ALERT_YAWN_COUNT         = 4
ALERT_COOLDOWN_SEC       = 5.0           # deduplication window per alert key
ALERT_HISTORY_MAXLEN     = 100

# ── Video capture ─────────────────────────────────────────────────────────────
CAP_DEVICE               = 0
CAP_WIDTH                = 640
CAP_HEIGHT               = 480
CAP_FPS                  = 30
CAP_BUFFER_SIZE          = 1             # 1-frame buffer eliminates webcam lag
CAP_FOURCC               = 'MJPG'

# ── MJPEG stream ──────────────────────────────────────────────────────────────
STREAM_JPEG_QUALITY      = 55

# ── Flask server ──────────────────────────────────────────────────────────────
FLASK_HOST               = '0.0.0.0'
FLASK_PORT               = 5003
FLASK_DEBUG              = False
FLASK_THREADED           = True

# ── Optical flow ──────────────────────────────────────────────────────────────
FLOW_PYR_SCALE           = 0.5
FLOW_LEVELS              = 2
FLOW_WINSIZE             = 10
FLOW_ITERATIONS          = 2
FLOW_POLY_N              = 5
FLOW_POLY_SIGMA          = 1.1

# ── Misc ──────────────────────────────────────────────────────────────────────
RPPG_SAMPLE_INTERVAL     = 3             # process rPPG every N frames
RESP_SAMPLE_INTERVAL     = 3             # process optical flow every N frames
STATS_POLL_MS            = 100           # frontend polling interval (ms)
ALERT_POLL_MS            = 2000          # frontend alert polling interval (ms)
CHART_MAX_POINTS         = 60            # data points kept in live chart
