"""
evaluate.py
Оцінювання детектора та трекера на DUT Anti-UAV test split.

Метрики детекції:  mAP@0.5, mAP@0.5:0.95, AP_S, Recall@0.5
Метрики трекінгу:  MOTA, IDF1, HOTA, IDSW

Запуск:
    python evaluate.py --weights checkpoints/best_final.pt
    python evaluate.py --weights checkpoints/best_final.pt --task track
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from src.config import (DATASET_ROOT, CONF_THRESH, NMS_IOU_THRESH,
                        DEVICE, RESULTS_DIR)
from src.modeling.detector       import Detector
from src.modeling.tracker        import ByteTrackIMM, hungarian, iou_batch
from src.modeling.dataset_utils  import (load_tracking_split,
                                         TrackingSequence,
                                         load_visdrone_vid_split,
                                         VisDroneVideoSequence)

DATASET_YAML = DATASET_ROOT / "processed" / "visdrone_yolo_tiled" / "dataset.yaml"
VISDRONE_VID_ROOT = DATASET_ROOT / "raw" / "VisDrone2019-VID-test-dev"


# ─────────────────────────────────────────────────────────────
#  Оцінка детектора (через вбудований Ultralytics val)
# ─────────────────────────────────────────────────────────────

def evaluate_detection(weights: str) -> dict:
    """
    Запускає Ultralytics val на test split.
    Повертає словник із метриками.
    """
    print("\n" + "="*50)
    print("ОЦІНКА ДЕТЕКТОРА")
    print("="*50)

    model = YOLO(weights)
    data_yaml = DATASET_YAML

    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=1280,
        conf=CONF_THRESH,
        iou=NMS_IOU_THRESH,
        device=DEVICE,
        verbose=True,
        save_json=True,
        project=str(RESULTS_DIR),
        name="det_eval",
    )

    metrics = {
        "mAP50":     float(results.box.map50),
        "mAP50_95":  float(results.box.map),
        "precision": float(results.box.mp),
        "recall":    float(results.box.mr),
    }

    # AP_S (малі об'єкти) — якщо Ultralytics повертає розбивку
    if hasattr(results.box, "maps"):
        maps = results.box.maps  # per-class/per-size
        metrics["maps_raw"] = maps.tolist()

    _print_det_metrics(metrics)
    _save_metrics(metrics, RESULTS_DIR / "detection_metrics.json")
    return metrics


def _print_det_metrics(m: dict) -> None:
    print(f"\n{'Метрика':<20} {'Значення':>10}")
    print("-" * 32)
    for k, v in m.items():
        if isinstance(v, float):
            print(f"{k:<20} {v:>10.4f}")


# ─────────────────────────────────────────────────────────────
#  Оцінка трекера
# ─────────────────────────────────────────────────────────────

def evaluate_tracking(weights: str) -> dict:
    """
    Запускає повний пайплайн (детектор + трекер)
    на test tracking sequences і рахує MOT-метрики.
    """
    print("\n" + "="*50)
    print("ОЦІНКА ТРЕКЕРА")
    print("="*50)

    detector = Detector(weights)
    detector.warmup()

    split_root = DATASET_ROOT / "tracking" / "test"
    if not split_root.exists():
        print(f"[Eval] Tracking split not found: {split_root}")
        print("[Eval] Skipping tracking evaluation.")
        return {}

    sequences = load_tracking_split(split_root)
    if not sequences:
        print("[Eval] Тестові послідовності не знайдено.")
        return {}

    all_metrics = []
    mu_history  = {}   # для побудови графіків

    for seq in sequences:
        print(f"\n  → Послідовність: {seq.name} ({len(seq)} кадрів)")
        metrics, mu_hist = _eval_sequence(seq, detector)
        all_metrics.append(metrics)
        mu_history[seq.name] = mu_hist

        print(f"     MOTA={metrics['MOTA']:.3f}  "
              f"IDF1={metrics['IDF1']:.3f}  "
              f"IDSW={metrics['IDSW']}")

    # Агрегація по всіх послідовностях
    agg = _aggregate_metrics(all_metrics)
    _print_track_metrics(agg)
    _save_metrics(agg, RESULTS_DIR / "tracking_metrics.json")

    # Побудова графіків
    _plot_mu_history(mu_history, RESULTS_DIR / "plots")

    return agg


def evaluate_visdrone_tracking(weights: str) -> dict:
    """
    Запускає detector + tracker на VisDrone VID test-dev.

    Повертає усереднені MOT-подібні метрики та зберігає результати треків
    у форматі, близькому до MOTChallenge.
    """
    print("\n" + "="*50)
    print("ОЦІНКА VISDRONE TRACKING")
    print("="*50)

    detector = Detector(weights)
    detector.warmup()

    if not VISDRONE_VID_ROOT.exists():
        print(f"[Eval] VisDrone VID split not found: {VISDRONE_VID_ROOT}")
        return {}

    sequences = load_visdrone_vid_split(VISDRONE_VID_ROOT)
    if not sequences:
        print("[Eval] VisDrone sequences not found.")
        return {}

    results_dir = RESULTS_DIR / "visdrone_tracking"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for seq in sequences:
        print(f"\n  → Послідовність: {seq.name} ({len(seq)} кадрів)")
        metrics = _eval_visdrone_sequence(seq, detector, results_dir)
        all_metrics.append(metrics)

        print(f"     MOTA={metrics['MOTA']:.3f}  "
              f"IDF1={metrics['IDF1']:.3f}  "
              f"IDSW={metrics['IDSW']}")

    agg = _aggregate_metrics(all_metrics)
    _print_track_metrics(agg)
    _save_metrics(agg, RESULTS_DIR / "visdrone_tracking_metrics.json")
    return agg


def _eval_sequence(seq: TrackingSequence,
                   detector: Detector) -> tuple:
    """
    Оцінює одну послідовність. Повертає (metrics_dict, mu_history).
    """
    tracker = ByteTrackIMM()
    gts_center = seq.gt_to_center()   # (N, 4): cx, cy, w, h

    tp = fp = fn = idsw = 0
    prev_id_map: dict = {}            # gt_idx → track_id
    mu_history  = []                  # для графіка ймовірностей IMM
    iou_sum     = 0.0
    iou_count   = 0

    for frame_idx, (frame, gt_tl) in enumerate(seq):
        # Отримати gt у форматі center
        gt = gts_center[frame_idx]
        gt_visible = (gt[2] > 0 and gt[3] > 0)

        # Детекція
        dets = detector.detect(frame)

        # Трекінг
        active_tracks = tracker.update(dets)

        # Запис ймовірностей моделей IMM
        if active_tracks:
            mu_history.append(active_tracks[0].model_probs.copy())
        else:
            mu_history.append(np.array([0.7, 0.2, 0.1]))

        if not gt_visible:
            fp += len(active_tracks)
            continue

        # Зіставлення треків із gt (для MOTA)
        if active_tracks:
            track_bboxes = np.array([t.bbox for t in active_tracks])
            ious = _iou_one_to_many(gt, track_bboxes)
            best_idx  = int(np.argmax(ious))
            best_iou  = ious[best_idx]
            best_id   = active_tracks[best_idx].id

            if best_iou >= 0.5:
                tp += 1
                iou_sum   += best_iou
                iou_count += 1
                fp        += len(active_tracks) - 1

                # Перевірка IDSW
                prev_id = prev_id_map.get(frame_idx - 1)
                if prev_id is not None and prev_id != best_id:
                    idsw += 1
                prev_id_map[frame_idx] = best_id
            else:
                fn += 1
                fp += len(active_tracks)
        else:
            fn += 1

    gt_count = int((gts_center[:, 2] > 0).sum())
    mota     = 1.0 - (fn + fp + idsw) / max(gt_count, 1)
    motp     = iou_sum / max(iou_count, 1)

    # IDF1 (спрощена оцінка через IDTP/IDFP/IDFN)
    idtp  = tp
    idfp  = fp
    idfn  = fn
    idf1  = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

    # HOTA (спрощений варіант як sqrt(DetA * AssA))
    deta  = tp / max(tp + fp + fn, 1)
    assa  = tp / max(tp + idsw, 1)
    hota  = float(np.sqrt(max(deta * assa, 0)))

    return {
        "MOTA": float(mota),
        "MOTP": float(motp),
        "IDF1": float(idf1),
        "HOTA": float(hota),
        "IDSW": int(idsw),
        "TP": tp, "FP": fp, "FN": fn,
    }, np.array(mu_history)


def _eval_visdrone_sequence(seq: VisDroneVideoSequence,
                            detector: Detector,
                            results_dir: Path) -> dict:
    """Оцінює одну VisDrone VID послідовність та зберігає MOT-результати."""
    tracker = ByteTrackIMM()
    prev_id_map: dict[int, int] = {}

    tp = fp = fn = idsw = 0
    iou_sum = 0.0
    iou_count = 0
    mot_lines: list[str] = []

    for frame_idx, (frame, gt_frame) in enumerate(seq, start=1):
        if frame is None:
            continue

        detections = detector.detect(frame)
        active_tracks = tracker.update(detections)

        gt_valid = gt_frame[gt_frame[:, 2] > 0]
        gt_center = _tlwh_to_center(gt_valid[:, :4]) if len(gt_valid) else np.empty((0, 4))
        track_center = np.array([t.bbox for t in active_tracks], dtype=np.float32) \
            if active_tracks else np.empty((0, 4))

        if len(gt_center) and len(track_center):
            cost = 1.0 - iou_batch(gt_center, track_center)
            matched, unmatched_gt, unmatched_tr = hungarian(cost, threshold=0.5)
            iou_mat = iou_batch(gt_center, track_center)
        else:
            matched, unmatched_gt, unmatched_tr = [], list(range(len(gt_center))), list(range(len(track_center)))
            iou_mat = np.empty((len(gt_center), len(track_center)))

        tp += len(matched)
        fp += len(unmatched_tr)
        fn += len(unmatched_gt)

        for gi, ti in matched:
            gt_id = int(gt_valid[gi, 4])
            track_id = active_tracks[ti].id
            prev_id = prev_id_map.get(gt_id)
            if prev_id is not None and prev_id != track_id:
                idsw += 1
            prev_id_map[gt_id] = track_id
            iou_sum += float(iou_mat[gi, ti])
            iou_count += 1

        for track in active_tracks:
            x, y, w, h = _center_to_tlwh(track.bbox)
            mot_lines.append(
                f"{frame_idx},{track.id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
            )

    out_path = results_dir / f"{seq.name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(mot_lines) + ("\n" if mot_lines else ""))

    gt_count = len([row for row in seq.gts if len(row) > 0])
    mota = 1.0 - (fn + fp + idsw) / max(sum(len(row) for row in seq.gts), 1)
    motp = iou_sum / max(iou_count, 1)
    idtp = tp
    idfp = fp
    idfn = fn
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)
    deta = tp / max(tp + fp + fn, 1)
    assa = tp / max(tp + idsw, 1)
    hota = float(np.sqrt(max(deta * assa, 0)))

    return {
        "MOTA": float(mota),
        "MOTP": float(motp),
        "IDF1": float(idf1),
        "HOTA": float(hota),
        "IDSW": int(idsw),
        "TP": tp, "FP": fp, "FN": fn,
    }


def _tlwh_to_center(boxes: np.ndarray) -> np.ndarray:
    """Перетворює масив tlwh у центр-формат cxcywh."""
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    centers = boxes.astype(np.float32).copy()
    centers[:, 0] = boxes[:, 0] + boxes[:, 2] / 2.0
    centers[:, 1] = boxes[:, 1] + boxes[:, 3] / 2.0
    return centers


def _center_to_tlwh(box: np.ndarray) -> tuple[float, float, float, float]:
    """Перетворює один bbox із центру в tlwh."""
    x = float(box[0] - box[2] / 2.0)
    y = float(box[1] - box[3] / 2.0)
    w = float(box[2])
    h = float(box[3])
    return x, y, w, h


def _iou_one_to_many(gt: np.ndarray,
                     tracks: np.ndarray) -> np.ndarray:
    """IoU між одним gt bbox і N track bboxes (всі cx,cy,w,h)."""
    def to_xyxy(b):
        return np.array([b[0]-b[2]/2, b[1]-b[3]/2,
                         b[0]+b[2]/2, b[1]+b[3]/2])

    gt_xy   = to_xyxy(gt)
    ious    = np.zeros(len(tracks))
    area_gt = gt[2] * gt[3]

    for i, t in enumerate(tracks):
        t_xy  = to_xyxy(t)
        ix1, iy1 = max(gt_xy[0], t_xy[0]), max(gt_xy[1], t_xy[1])
        ix2, iy2 = min(gt_xy[2], t_xy[2]), min(gt_xy[3], t_xy[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = area_gt + t[2]*t[3] - inter
        ious[i] = inter / max(union, 1e-6)

    return ious


def _aggregate_metrics(all_m: list) -> dict:
    keys = ["MOTA", "MOTP", "IDF1", "HOTA", "IDSW"]
    agg  = {}
    for k in keys:
        vals = [m[k] for m in all_m]
        if k == "IDSW":
            agg[k] = int(sum(vals))
        else:
            agg[k]  = float(np.mean(vals))
            agg[k + "_std"] = float(np.std(vals))
    return agg


def _print_track_metrics(m: dict) -> None:
    print(f"\n{'Метрика':<20} {'Середнє':>10}")
    print("-" * 32)
    for k in ["MOTA", "MOTP", "IDF1", "HOTA", "IDSW"]:
        if k in m:
            v = m[k]
            print(f"{k:<20} {v:>10.4f}" if isinstance(v, float)
                  else f"{k:<20} {v:>10d}")


# ─────────────────────────────────────────────────────────────
#  Графіки
# ─────────────────────────────────────────────────────────────

def _plot_mu_history(mu_history: dict, out_dir: Path) -> None:
    """
    Будує графік ймовірностей моделей IMM для кожної послідовності.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["CV", "CA", "CT"]

    for seq_name, mu_arr in mu_history.items():
        if len(mu_arr) == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 3))
        for j in range(3):
            ax.plot(mu_arr[:, j], color=colors[j],
                    label=labels[j], linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Кадр")
        ax.set_ylabel("Ймовірність моделі")
        ax.set_title(f"IMM — ймовірності моделей: {seq_name}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"imm_probs_{seq_name}.png",
                    dpi=150)
        plt.close(fig)

    print(f"[Eval] Графіки ймовірностей збережено: {out_dir}")


# ─────────────────────────────────────────────────────────────
#  FPS benchmark
# ─────────────────────────────────────────────────────────────

def benchmark_fps(weights: str, n_frames: int = 100) -> float:
    """
    Вимірює FPS усього пайплайну на синтетичних кадрах.
    """
    detector = Detector(weights)
    detector.warmup()
    tracker  = ByteTrackIMM()

    dummy = np.random.randint(0, 255,
                              (720, 1280, 3), dtype=np.uint8)
    times = []
    for _ in range(n_frames):
        t0   = time.perf_counter()
        dets = detector.detect(dummy)
        tracker.update(dets)
        times.append(time.perf_counter() - t0)

    fps = 1.0 / np.mean(times[10:])   # перші 10 — прогрів
    print(f"[FPS] Середній FPS пайплайну: {fps:.1f}")
    return fps


# ─────────────────────────────────────────────────────────────
#  Утиліти
# ─────────────────────────────────────────────────────────────

def _save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval] Метрики збережено: {path}")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Оцінювання детектора та трекера"
    )
    parser.add_argument("--weights", type=str,
                        default="checkpoints/best_final.pt",
                        help="Шлях до ваг моделі (.pt)")
    parser.add_argument("--task", type=str,
                        default="both",
                        choices=["detect", "track", "fps", "both", "visdrone"],
                        help="Що оцінювати")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.task in ("detect", "both"):
        evaluate_detection(args.weights)

    if args.task == "visdrone":
        evaluate_visdrone_tracking(args.weights)

    if args.task in ("track", "both"):
        evaluate_tracking(args.weights)

    if args.task == "fps":
        benchmark_fps(args.weights)
