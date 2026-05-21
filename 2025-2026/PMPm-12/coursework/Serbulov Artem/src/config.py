import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


# ---------------------------------------------------------------------------
# Defaults and runtime constants used by evaluation / tracking / inference
# These provide sensible defaults so scripts like `evaluate_det_track.py`
# and `src/modeling/*` can import a single canonical config.
# ---------------------------------------------------------------------------

# Dataset / output paths (can be overridden via env or edited by user)
DATASET_ROOT = PROJ_ROOT / "data"
RESULTS_DIR = PROJ_ROOT / "results"

# Inference / detection defaults
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.5))
NMS_IOU_THRESH = float(os.getenv("NMS_IOU_THRESH", 0.5))
MAX_DETS = int(os.getenv("MAX_DETS", 300))
# Inference image size used by Detector (height, width)
INFER_IMG_SIZE = (640, 640)
DEVICE = os.getenv("DEVICE", "0")

# Tracking / ByteTrack-IMM defaults
TAU_HIGH = float(os.getenv("TAU_HIGH", 0.6))
TAU_LOW = float(os.getenv("TAU_LOW", 0.3))
TAU_NEW = float(os.getenv("TAU_NEW", 0.7))
MAX_AGE = int(os.getenv("MAX_AGE", 30))
N_INIT = int(os.getenv("N_INIT", 3))
COST_ALPHA = float(os.getenv("COST_ALPHA", 0.9))
# Chi-square gating threshold (e.g., 3 DOF -> 7.81, 4 DOF -> 9.21)
GATE_CHI2 = float(os.getenv("GATE_CHI2", 9.21))
DT = float(os.getenv("DT", 1.0))

