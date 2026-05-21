"""
train.py
Навчання всіх конфігурацій ablation study.

Конфігурації:
  b1 — YOLOv8s стандартний         (yolov8s.yaml,    без SimAM, CIoU)
  b2 — YOLOv8s + P2                (yolov8s_p2.yaml, без SimAM, CIoU)
  b3 — YOLOv8s + P2 + SimAM        (yolov8s_p2.yaml, SimAM,    CIoU)
  b4 — YOLOv8s + P2 + SimAM + WIoU (yolov8s_p2.yaml, SimAM,    WiseIoU)
  p  — те саме що b4 (трекер різний, детектор однаковий)
  all— навчити b1 → b2 → b3 → b4 послідовно

Запуск:
  python train.py --config b1          # лише baseline
  python train.py --config b4          # лише proposed detector
  python train.py --config all         # всі конфігурації
  python train.py --config b1 --stage 1   # лише 1-й етап для b1
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from config import (
    DATASET_ROOT, PRETRAINED, MODEL_YAML,
    TRAIN_IMG_SIZE, EPOCHS_STAGE1, EPOCHS_STAGE2,
    BATCH_SIZE, LR_STAGE1, LR_STAGE2,
    WEIGHT_DECAY, EARLY_STOP_PATIENCE,
    WORKERS, DEVICE, CHECKPOINTS_DIR,
)
from dataset_utils import make_data_yaml
from wiseiou       import WiseIoUTrainer


# ─────────────────────────────────────────────────────────────
#  Визначення конфігурацій
# ─────────────────────────────────────────────────────────────

# Для кожної конфігурації: яку YAML використовувати,
# чи вставляти SimAM hooks, чи використовувати WiseIoU тренер
CONFIG_DEFS = {
    "b1": dict(
        yaml     = "yolov8s.yaml",   # стандартна Ultralytics
        use_simam = False,
        use_wiseiou = False,
        desc     = "YOLOv8s стандартний (baseline)",
    ),
    "b2": dict(
        yaml      = str(MODEL_YAML),
        use_simam  = False,
        use_wiseiou = False,
        desc      = "YOLOv8s + голова P2",
    ),
    "b3": dict(
        yaml      = str(MODEL_YAML),
        use_simam  = True,
        use_wiseiou = False,
        desc      = "YOLOv8s + P2 + SimAM",
    ),
    "b4": dict(
        yaml      = str(MODEL_YAML),
        use_simam  = True,
        use_wiseiou = True,
        desc      = "YOLOv8s + P2 + SimAM + Wise-IoU  (= P detector)",
    ),
}
# p і b4 мають однаковий детектор — окремо не тренуємо
CONFIG_DEFS["p"] = CONFIG_DEFS["b4"]


# ─────────────────────────────────────────────────────────────
#  Спільні параметри навчання
# ─────────────────────────────────────────────────────────────

def _base_overrides(cfg_name: str, stage: int,
                    yaml: str, data_yaml: Path,
                    weights: str) -> dict:
    """Будує словник overrides для Ultralytics Trainer."""
    is_stage2 = (stage == 2)
    return {
        "model":       weights if is_stage2 else yaml,
        "data":        str(data_yaml),
        "epochs":      EPOCHS_STAGE2 if is_stage2 else EPOCHS_STAGE1,
        "imgsz":       TRAIN_IMG_SIZE,
        "batch":       BATCH_SIZE,
        "lr0":         LR_STAGE2 if is_stage2 else LR_STAGE1,
        "weight_decay": WEIGHT_DECAY,
        "optimizer":   "SGD" if is_stage2 else "AdamW",
        "patience":    EARLY_STOP_PATIENCE,
        "workers":     WORKERS,
        "device":      DEVICE,
        "project":     "runs/detect",
        "name":        f"{cfg_name}_stage{stage}",
        "save":        True,
        "pretrained":  PRETRAINED if not is_stage2 else False,
        "freeze":      10 if is_stage2 else 0,
        # Аугментація
        "mosaic":      0.0 if is_stage2 else 1.0,
        "mixup":       0.0 if is_stage2 else 0.1,
        "fliplr":      0.5,
        "degrees":     5.0 if is_stage2 else 10.0,
        "translate":   0.05 if is_stage2 else 0.1,
        "scale":       0.3 if is_stage2 else 0.5,
        "hsv_h":       0.015,
        "hsv_s":       0.7,
        "hsv_v":       0.4,
    }


# ─────────────────────────────────────────────────────────────
#  Вибір тренера
# ─────────────────────────────────────────────────────────────

def _get_trainer(use_simam: bool, use_wiseiou: bool,
                 overrides: dict):
    """
    Повертає правильний клас тренера залежно від конфігурації.
    SimAM вставляється через build_model після ініціалізації,
    якщо потрібен.
    """
    TrainerClass = WiseIoUTrainer if use_wiseiou \
                   else DetectionTrainer
    trainer = TrainerClass(overrides=overrides)

    # Якщо SimAM потрібен — патчимо get_model тренера
    if use_simam:
        _inject_simam(trainer)

    return trainer


def _inject_simam(trainer) -> None:
    """
    Обгортає get_model тренера так, щоб після побудови моделі
    автоматично вставлялись SimAM hooks.
    """
    from model_builder import _attach_simam_hooks
    original_get_model = trainer.__class__.get_model

    def patched_get_model(self, cfg=None, weights=None, verbose=True):
        model = original_get_model(self, cfg, weights, verbose)
        _attach_simam_hooks(model)
        print("[Train] SimAM hooks встановлено.")
        return model

    # Прив'язуємо до конкретного екземпляра
    import types
    trainer.get_model = types.MethodType(patched_get_model, trainer)


# ─────────────────────────────────────────────────────────────
#  Навчання однієї конфігурації
# ─────────────────────────────────────────────────────────────

def train_config(cfg_name: str, data_yaml: Path,
                 stage: int = 0) -> Path:
    """
    Навчає одну конфігурацію в один або два етапи.

    Args:
        cfg_name:  'b1' | 'b2' | 'b3' | 'b4' | 'p'
        data_yaml: шлях до data.yaml
        stage:     0 = обидва, 1 = лише Етап I, 2 = лише Етап II

    Returns:
        шлях до збережених фінальних ваг
    """
    cfg = CONFIG_DEFS[cfg_name]
    print("\n" + "="*60)
    print(f"КОНФІГУРАЦІЯ {cfg_name.upper()}: {cfg['desc']}")
    print("="*60)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Етап I ────────────────────────────────────────────────
    if stage in (0, 1):
        ov1 = _base_overrides(cfg_name, 1, cfg["yaml"],
                              data_yaml, cfg["yaml"])
        trainer1 = _get_trainer(cfg["use_simam"],
                                cfg["use_wiseiou"], ov1)
        trainer1.train()
        s1_weights = Path(trainer1.best)
        print(f"[{cfg_name}] Етап I → {s1_weights}")
    else:
        # Якщо пропускаємо Етап I — беремо вже існуючі ваги
        s1_weights = Path(
            f"runs/detect/{cfg_name}_stage1/weights/best.pt"
        )
        if not s1_weights.exists():
            raise FileNotFoundError(
                f"Ваги Етапу I не знайдено: {s1_weights}\n"
                f"Запустіть спочатку: python train.py "
                f"--config {cfg_name} --stage 1"
            )

    # ── Етап II ───────────────────────────────────────────────
    if stage in (0, 2):
        ov2 = _base_overrides(cfg_name, 2, cfg["yaml"],
                              data_yaml, str(s1_weights))
        trainer2 = _get_trainer(cfg["use_simam"],
                                cfg["use_wiseiou"], ov2)
        trainer2.train()
        best = Path(trainer2.best)
    else:
        best = s1_weights

    # Копіюємо у checkpoints/
    dst = CHECKPOINTS_DIR / f"{cfg_name}_best.pt"
    shutil.copy(best, dst)
    print(f"[{cfg_name}] Фінальні ваги → {dst}")
    return dst


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main(args):
    data_yaml = DATASET_ROOT / "data.yaml"
    if not data_yaml.exists():
        make_data_yaml(DATASET_ROOT, data_yaml)

    if args.config == "all":
        # Навчаємо B1 → B2 → B3 → B4 послідовно
        # P пропускаємо — той самий детектор що B4
        results = {}
        for cfg_name in ["b1", "b2", "b3", "b4"]:
            dst = train_config(cfg_name, data_yaml, stage=args.stage)
            results[cfg_name] = dst

        print("\n" + "="*60)
        print("Навчання всіх конфігурацій завершено:")
        for k, v in results.items():
            print(f"  {k.upper()} → {v}")
        print("="*60)
        print("\nДля ablation study запустіть:")
        print("  python ablation.py \\")
        for k, v in results.items():
            print(f"    --weights_{k} {v} \\")
        print("    --weights_p checkpoints/b4_best.pt")

    else:
        cfg_name = args.config.lower()
        if cfg_name not in CONFIG_DEFS:
            raise ValueError(
                f"Невідома конфігурація '{cfg_name}'. "
                f"Доступні: {list(CONFIG_DEFS.keys())} або 'all'"
            )
        train_config(cfg_name, data_yaml, stage=args.stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Навчання конфігурацій YOLOv8s для ablation study"
    )
    parser.add_argument(
        "--config", type=str, default="b4",
        help="b1 | b2 | b3 | b4 | p | all"
    )
    parser.add_argument(
        "--stage", type=int, default=0,
        help="0=обидва етапи, 1=лише Етап I, 2=лише Етап II"
    )
    args = parser.parse_args()
    main(args)
