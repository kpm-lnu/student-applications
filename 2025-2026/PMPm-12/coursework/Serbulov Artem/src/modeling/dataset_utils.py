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
from collections import defaultdict
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


class VisDroneVideoSequence:
    """
    Читає одну VisDrone VID послідовність.

    Формат:
        sequences/<seq_name>/*.jpg
        annotations/<seq_name>.txt

    Для кожного кадру зберігає масив GT-об'єктів з колонками:
        x_tl, y_tl, w, h, object_id, class_id
    """

    def __init__(self, seq_dir: Path, ann_path: Path):
        self.name = seq_dir.name
        self.seq_dir = seq_dir

        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        self.frames = sorted([
            p for p in seq_dir.iterdir()
            if p.suffix.lower() in exts
        ])

        self.gts = self._load_gt(ann_path)

    def _load_gt(self, path: Path) -> List[np.ndarray]:
        """Читає VisDrone annotation файл у frame-wise формат."""
        frame_map: dict[int, list[list[float]]] = defaultdict(list)

        if not path.exists():
            print(f"[VisDroneSeq] gt не знайдено: {path}")
            return [np.zeros((0, 6), dtype=np.float32)
                    for _ in range(len(self.frames))]

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                vals = line.replace(",", " ").split()
                if len(vals) < 6:
                    continue

                try:
                    frame_id = int(float(vals[0]))
                    object_id = int(float(vals[1]))
                    x_tl = float(vals[2])
                    y_tl = float(vals[3])
                    width = float(vals[4])
                    height = float(vals[5])
                    score = float(vals[6]) if len(vals) > 6 else 1.0
                    class_id = int(float(vals[7])) if len(vals) > 7 else -1
                except ValueError:
                    continue

                if width <= 0 or height <= 0 or score <= 0:
                    continue

                frame_map[frame_id].append(
                    [x_tl, y_tl, width, height, object_id, class_id]
                )

        gts = [np.zeros((0, 6), dtype=np.float32)
               for _ in range(len(self.frames))]
        for frame_id, boxes in frame_map.items():
            if 1 <= frame_id <= len(self.frames):
                gts[frame_id - 1] = np.array(boxes, dtype=np.float32)

        return gts

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Ітерує (frame_bgr, gt_boxes) парами."""
        for img_path, gt in zip(self.frames, self.gts):
            frame = cv2.imread(str(img_path))
            yield frame, gt


def load_visdrone_vid_split(split_root: Path) -> List[VisDroneVideoSequence]:
    """
    Завантажує всі VisDrone VID послідовності зі split_root.

    Очікує структуру:
        split_root/
          sequences/<seq_name>/*.jpg
          annotations/<seq_name>.txt
    """
    seq_root = split_root / "sequences"
    ann_root = split_root / "annotations"

    if not seq_root.exists():
        return []

    sequences: List[VisDroneVideoSequence] = []
    for seq_dir in sorted(seq_root.iterdir()):
        if seq_dir.is_dir():
            sequences.append(
                VisDroneVideoSequence(seq_dir, ann_root / f"{seq_dir.name}.txt")
            )

    print(f"[DataUtils] Завантажено {len(sequences)} VisDrone послідовностей ")
    return sequences


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
