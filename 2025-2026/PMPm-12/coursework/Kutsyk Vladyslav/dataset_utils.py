"""
dataset_utils.py
Утиліти для роботи з датасетом DUT Anti-UAV.

Структура датасету на диску:
    data/DUT-Anti-UAV/
      detection/
        train/images/*.jpg
        train/labels/*.txt     ← YOLO format: class cx cy w h (normalized)
        val/images/   val/labels/
        test/images/  test/labels/
      tracking/
        train/<seq_name>/  val/<seq_name>/  test/<seq_name>/
          imgs/  *.jpg
          groundtruth_rect.txt   ← кожен рядок: x_tl y_tl w h (пікселі)
          attr.txt               (опційно)

Формат groundtruth_rect.txt (OTB-style):
    Кожен рядок = один кадр: x_topleft y_topleft width height
    Значення розділені пробілами або комами.
    Якщо ціль невидима — значення можуть бути 0 0 0 0.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Iterator, Optional
import yaml


# ─────────────────────────────────────────────────────────────
#  Генерація data.yaml для Ultralytics
# ─────────────────────────────────────────────────────────────

def make_data_yaml(dataset_root: Path,
                   output_path:  Path,
                   nc: int = 1,
                   names: List[str] = None) -> Path:
    """
    Генерує data.yaml у форматі Ultralytics для detection split.
    Потрібен для trainer.train(data=...).

    Args:
        dataset_root: корінь DUT-Anti-UAV (де лежать detection/)
        output_path:  куди записати data.yaml
        nc:           кількість класів
        names:        список назв класів
    """
    if names is None:
        names = ["drone"]

    det_root = dataset_root / "detection"

    config = {
        "path":  str(dataset_root.resolve()),
        "train": str((det_root / "train" / "images").relative_to(
                     dataset_root.resolve())),
        "val":   str((det_root / "val"   / "images").relative_to(
                     dataset_root.resolve())),
        "test":  str((det_root / "test"  / "images").relative_to(
                     dataset_root.resolve())),
        "nc":    nc,
        "names": names,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"[DataUtils] data.yaml збережено: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────
#  Tracking sequence reader
# ─────────────────────────────────────────────────────────────

class TrackingSequence:
    """
    Читає одну відеопослідовність DUT Anti-UAV.

    Атрибути:
        name:    назва послідовності (ім'я папки)
        frames:  список шляхів до зображень (відсортованих)
        gts:     np.ndarray (N, 4) — x_tl, y_tl, w, h у пікселях
                 (0,0,0,0) = ціль невидима
    """

    def __init__(self, seq_dir: Path):
        self.name    = seq_dir.name
        self.seq_dir = seq_dir

        # Зображення
        imgs_dir = seq_dir / "imgs"
        if not imgs_dir.exists():
            imgs_dir = seq_dir  # на випадок іншої структури

        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        self.frames = sorted([
            p for p in imgs_dir.iterdir()
            if p.suffix.lower() in exts
        ])

        # Анотації
        gt_path = seq_dir / "groundtruth_rect.txt"
        self.gts = self._load_gt(gt_path)

        if len(self.frames) != len(self.gts):
            print(f"[TrackSeq] ПОПЕРЕДЖЕННЯ: {self.name} — "
                  f"кадрів {len(self.frames)}, "
                  f"gt рядків {len(self.gts)}")

    def _load_gt(self, path: Path) -> np.ndarray:
        """
        Зчитує groundtruth_rect.txt.
        Підтримує розділювачі: пробіл, кома, таб.
        Формат кожного рядка: x_tl y_tl w h
        """
        if not path.exists():
            print(f"[TrackSeq] gt не знайдено: {path}")
            return np.zeros((len(self.frames), 4))

        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Підтримка коми або пробілу як розділювача
                vals = line.replace(",", " ").split()
                try:
                    rows.append([float(v) for v in vals[:4]])
                except ValueError:
                    rows.append([0., 0., 0., 0.])

        return np.array(rows, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Ітерує (frame_bgr, gt_bbox) парами."""
        for img_path, gt in zip(self.frames, self.gts):
            frame = cv2.imread(str(img_path))
            yield frame, gt

    def gt_to_center(self) -> np.ndarray:
        """
        Повертає gt у форматі [cx, cy, w, h]:
        корисно для порівняння з виходом детектора.
        """
        gts_c = self.gts.copy()
        gts_c[:, 0] = self.gts[:, 0] + self.gts[:, 2] / 2.0  # cx
        gts_c[:, 1] = self.gts[:, 1] + self.gts[:, 3] / 2.0  # cy
        return gts_c


def load_tracking_split(split_dir: Path) -> List[TrackingSequence]:
    """
    Завантажує всі послідовності з папки split_dir.
    Очікує структуру: split_dir/<seq_name>/
    """
    sequences = []
    for seq_dir in sorted(split_dir.iterdir()):
        if seq_dir.is_dir():
            sequences.append(TrackingSequence(seq_dir))

    print(f"[DataUtils] Завантажено {len(sequences)} послідовностей "
          f"із {split_dir}")
    return sequences


# ─────────────────────────────────────────────────────────────
#  Статистика датасету
# ─────────────────────────────────────────────────────────────

def detection_split_stats(det_root: Path) -> dict:
    """
    Рахує статистику detection split:
    кількість зображень, розподіл площ bbox.
    """
    stats = {split: {"images": 0, "labels": 0,
                     "areas": []}
             for split in ["train", "val", "test"]}

    for split in stats:
        lbl_dir = det_root / split / "labels"
        img_dir = det_root / split / "images"

        if img_dir.exists():
            stats[split]["images"] = len(list(img_dir.glob("*.jpg")))

        if lbl_dir.exists():
            label_files = list(lbl_dir.glob("*.txt"))
            stats[split]["labels"] = len(label_files)

            for lf in label_files:
                with open(lf) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            # YOLO normalized: cx cy w h
                            w, h = float(parts[3]), float(parts[4])
                            # Площа у відносних одиницях
                            stats[split]["areas"].append(w * h)

    return stats
