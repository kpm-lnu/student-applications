"""
config.py
Усі гіперпараметри та шляхи проєкту в одному місці.
Змінюйте лише цей файл — рештa коду читає звідси.
"""

from pathlib import Path

# ── ШЛЯХИ ─────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parents[2]
DATASET_ROOT     = PROJECT_ROOT / "src" / "modeling" / "data" / "DUT-Anti-UAV"
DET_TRAIN_DIR    = DATASET_ROOT / "detection" / "train"
DET_VAL_DIR      = DATASET_ROOT / "detection" / "val"
DET_TEST_DIR     = DATASET_ROOT / "detection" / "test"
TRACK_TRAIN_DIR  = DATASET_ROOT / "tracking" / "train"
TRACK_VAL_DIR    = DATASET_ROOT / "tracking" / "val"
TRACK_TEST_DIR   = DATASET_ROOT / "tracking" / "test"

CHECKPOINTS_DIR  = PROJECT_ROOT / "checkpoints"
RESULTS_DIR      = PROJECT_ROOT / "results"
MODEL_YAML       = PROJECT_ROOT / "src" / "modeling" / "yolov8s_p2.yaml"

# ── МОДЕЛЬ ────────────────────────────────────────────────────
NUM_CLASSES     = 1
CLASS_NAMES     = ["drone"]
PRETRAINED      = "yolov8s.pt"   # базові ваги для трансферного навчання

# ── НАВЧАННЯ ──────────────────────────────────────────────────
TRAIN_IMG_SIZE       = 640
EPOCHS_STAGE1        = 10        # навчання на DUT detection split
EPOCHS_STAGE2        = 10       # дофайнтюнінг з замороженим backbone
#BATCH_SIZE           = 16 b1
BATCH_SIZE           = 8 # b4
LR_STAGE1            = 1e-3
LR_STAGE2            = 1e-4
WEIGHT_DECAY         = 5e-4
EARLY_STOP_PATIENCE  = 15       # епох без покращення mAP0.5
WORKERS              = 0
DEVICE               = "cuda"   # "cpu" якщо GPU недоступний

# ── ІНФЕРЕНС ──────────────────────────────────────────────────
INFER_IMG_SIZE    = (1280, 720)  # (W, H) для відео
CONF_THRESH       = 0.25
NMS_IOU_THRESH    = 0.45
MAX_DETS          = 300

# ── ТРЕКЕР ────────────────────────────────────────────────────
TAU_HIGH          = 0.60    # поріг впевненості для 1-го проходу
TAU_LOW           = 0.10    # поріг впевненості для 2-го проходу
TAU_NEW           = 0.60    # поріг для ініціації нового треку
MAX_AGE           = 30      # макс. кадрів без детекції до видалення треку
N_INIT            = 3       # кадрів для підтвердження нового треку
COST_ALPHA        = 0.70    # вага Mahal у комбінованій вартості
GATE_CHI2         = 9.488   # chi2(4, 0.95) — поріг воріт Махаланобіса

# ── IMM-KF ────────────────────────────────────────────────────
FPS               = 25.0
DT                = 1.0 / FPS

# Початковий розподіл ймовірностей моделей [CV, CA, CT]
IMM_MU_INIT       = [0.70, 0.20, 0.10]

# Матриця переходів між моделями (рядок = з, стовпець = в)
IMM_PI = [
    [0.80, 0.15, 0.05],   # з CV → CV, CA, CT
    [0.15, 0.75, 0.10],   # з CA
    [0.05, 0.20, 0.75],   # з CT
]

# Шуми процесу (дисперсія) для кожної моделі — задаються вручну
# або налаштовуються через evaluate_R() в imm_kalman.py
NOISE_CV = dict(pos=1.0, vel=5.0, acc=0.0,  wh=0.5)
NOISE_CA = dict(pos=1.0, vel=5.0, acc=20.0, wh=0.5)
NOISE_CT = dict(pos=1.0, vel=5.0, acc=0.0,  wh=0.5)

# Шум вимірювання (дисперсія у пікселях для x, y, w, h)
MEAS_NOISE = [2.0, 2.0, 5.0, 5.0]

# ── SimAM ─────────────────────────────────────────────────────
SIMAM_LAMBDA      = 1e-4

# ── Wise-IoU v3 ───────────────────────────────────────────────
WISEIOU_BETA      = 2.0
WISEIOU_DELTA     = 3.0
