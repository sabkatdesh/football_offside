import os
from pathlib import Path

# ── Base directories ───────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent.parent  # project root
DATA_DIR   = BASE_DIR / "data"

# Allow env var overrides (used in Modal deployment)
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", str(DATA_DIR / "uploads")))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(DATA_DIR / "outputs")))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(DATA_DIR / "models")))

# ── Model paths ────────────────────────────────────────────────────────────────
PITCH_MODEL_PATH  = MODELS_DIR / "football-pitch-detection.pt"
PLAYER_MODEL_PATH = MODELS_DIR / "football-player-detection.pt"

# ── Detection constants ────────────────────────────────────────────────────────
BALL_ID       = 0
GOALKEEPER_ID = 1
PLAYER_ID     = 2
REFEREE_ID    = 3

DETECTION_CONF    = 0.3
NMS_THRESHOLD     = 0.5
KP_CONF_THRESHOLD = 0.5
STRIDE            = 30

# ── Radar ─────────────────────────────────────────────────────────────────────
RADAR_SCALE   = 0.065
RADAR_PADDING = 20

# ── Ball trail ────────────────────────────────────────────────────────────────
BALL_TRAIL_LEN    = 40
HOMOGRAPHY_SMOOTH = 5

# ── API ───────────────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 500
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# ── Ensure dirs exist ──────────────────────────────────────────────────────────
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)