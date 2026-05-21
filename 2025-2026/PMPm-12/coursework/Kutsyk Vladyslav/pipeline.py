"""
pipeline.py
Повний пайплайн виявлення та трекінгу для відео або зображень.

Запуск на відео:
    python pipeline.py --weights checkpoints/best_final.pt \
                       --source data/DUT-Anti-UAV/tracking/test/seq1/imgs

Запуск на вебкамері:
    python pipeline.py --weights checkpoints/best_final.pt --source 0

Виведення:
  - Відео з bounding boxes та ID треків
  - Графік ймовірностей IMM у реальному часі (опційно --show_imm)
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from config       import RESULTS_DIR, INFER_IMG_SIZE
from detector     import Detector
from tracker      import ByteTrackIMM, Track


# ─────────────────────────────────────────────────────────────
#  Кольори та шрифт для візуалізації
# ─────────────────────────────────────────────────────────────
PALETTE = [
    (0,   200, 100),   # зелений  (confirmed)
    (50,  150, 255),   # блакитний
    (255, 100,  50),   # помаранчевий
    (200,  50, 200),   # фіолетовий
]

FONT  = cv2.FONT_HERSHEY_SIMPLEX
THICK = 2


def get_color(track_id: int) -> tuple:
    return PALETTE[track_id % len(PALETTE)]


# ─────────────────────────────────────────────────────────────
#  Основний клас пайплайну
# ─────────────────────────────────────────────────────────────

class UAVPipeline:
    """
    Клас, що об'єднує детектор і трекер у єдиний потік обробки.
    """

    def __init__(self,
                 weights: str,
                 show_imm: bool = False):
        self.detector  = Detector(weights)
        self.tracker   = ByteTrackIMM()
        self.show_imm  = show_imm
        self.frame_idx = 0
        self.times     = []

        # Буфер ймовірностей для живого графіка
        self._mu_buf: list = []

        self.detector.warmup()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Обробляє один кадр.
        Повертає кадр із намальованими bounding boxes та ID.
        """
        t0 = time.perf_counter()

        # ── Детекція ──────────────────────────────────────────
        dets = self.detector.detect(frame)

        # ── Трекінг ───────────────────────────────────────────
        tracks = self.tracker.update(dets)

        # ── Запис часу ────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        self.times.append(elapsed)

        # ── Збір IMM ймовірностей ─────────────────────────────
        if tracks:
            self._mu_buf.append(tracks[0].model_probs.copy())

        # ── Візуалізація ──────────────────────────────────────
        vis = self._draw(frame.copy(), tracks, dets)
        self.frame_idx += 1
        return vis

    def _draw(self, frame: np.ndarray,
              tracks: list,
              dets:   np.ndarray) -> np.ndarray:
        """Малює bounding boxes детекцій та треків."""

        h, w = frame.shape[:2]

        # Детекції (маленькі сірі рамки)
        for det in dets:
            cx, cy, bw, bh = det[:4]
            x1 = int(cx - bw/2); y1 = int(cy - bh/2)
            x2 = int(cx + bw/2); y2 = int(cy + bh/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (120, 120, 120), 1)

        # Треки (кольорові рамки з ID)
        for t in tracks:
            cx, cy, bw, bh = t.bbox
            x1 = int(cx - bw/2); y1 = int(cy - bh/2)
            x2 = int(cx + bw/2); y2 = int(cy + bh/2)
            color = get_color(t.id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICK)

            # ID + домінуюча модель IMM
            mu = t.model_probs
            model_name = ["CV", "CA", "CT"][int(np.argmax(mu))]
            label = f"ID:{t.id} [{model_name} {mu.max():.2f}]"

            lbl_y = max(y1 - 6, 14)
            cv2.putText(frame, label,
                        (x1, lbl_y), FONT, 0.45,
                        color, 1, cv2.LINE_AA)

        # FPS у лівому верхньому куті
        if len(self.times) > 5:
            fps = 1.0 / np.mean(self.times[-20:])
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (8, 24), FONT, 0.65,
                        (0, 255, 0), 2, cv2.LINE_AA)

        # Кількість активних треків
        cv2.putText(frame, f"Треків: {len(tracks)}",
                    (8, 50), FONT, 0.55,
                    (0, 230, 230), 1, cv2.LINE_AA)

        return frame

    def avg_fps(self) -> float:
        if len(self.times) < 5:
            return 0.0
        return 1.0 / np.mean(self.times[5:])


# ─────────────────────────────────────────────────────────────
#  Читання джерела (відео, папка зображень, камера)
# ─────────────────────────────────────────────────────────────

def open_source(source: str):
    """
    Повертає генератор кадрів (np.ndarray BGR).
    source: шлях до відео, папки з зображеннями або ціле число (камера).
    """
    # Числовий індекс камери
    try:
        cam_id = int(source)
        cap = cv2.VideoCapture(cam_id)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()
        return
    except ValueError:
        pass

    src_path = Path(source)

    # Папка із зображеннями
    if src_path.is_dir():
        img_paths = sorted(src_path.glob("*.jpg")) + \
                    sorted(src_path.glob("*.png"))
        for p in img_paths:
            frame = cv2.imread(str(p))
            if frame is not None:
                yield frame
        return

    # Відеофайл
    if src_path.is_file():
        cap = cv2.VideoCapture(str(src_path))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()
        return

    raise ValueError(f"Невідоме джерело: {source}")


# ─────────────────────────────────────────────────────────────
#  Запуск
# ─────────────────────────────────────────────────────────────

def run(args):
    pipeline = UAVPipeline(
        weights=args.weights,
        show_imm=args.show_imm,
    )

    # Налаштування виводу у відеофайл
    writer: Optional[cv2.VideoWriter] = None
    if args.save_video:
        out_path = RESULTS_DIR / "output.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 25,
                                 INFER_IMG_SIZE)

    try:
        for frame in open_source(args.source):
            vis = pipeline.process_frame(frame)

            if args.show:
                cv2.imshow("UAV Detection & Tracking", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer is not None:
                resized = cv2.resize(vis, INFER_IMG_SIZE)
                writer.write(resized)

    finally:
        cv2.destroyAllWindows()
        if writer:
            writer.release()

    print(f"\n[Pipeline] Оброблено кадрів: {pipeline.frame_idx}")
    print(f"[Pipeline] Середній FPS: {pipeline.avg_fps():.1f}")

    # Збереження графіка ймовірностей IMM
    if pipeline._mu_buf and args.show_imm:
        mu_arr = np.array(pipeline._mu_buf)
        fig, ax = plt.subplots(figsize=(12, 3))
        for j, (lbl, col) in enumerate(
                zip(["CV", "CA", "CT"],
                    ["tab:blue", "tab:orange", "tab:green"])):
            ax.plot(mu_arr[:, j], label=lbl, color=col, lw=1.5)
        ax.set_ylim(0, 1); ax.set_xlabel("Кадр")
        ax.set_ylabel("Ймовірність"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        imm_path = RESULTS_DIR / "imm_probs_live.png"
        fig.savefig(imm_path, dpi=150)
        print(f"[Pipeline] IMM-графік: {imm_path}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    type=str,
                        default="checkpoints/best_final.pt")
    parser.add_argument("--source",     type=str,
                        default="0",
                        help="Відео / папка зображень / ID камери")
    parser.add_argument("--show",       action="store_true",
                        help="Показувати вікно OpenCV")
    parser.add_argument("--save_video", action="store_true",
                        help="Зберегти results/output.mp4")
    parser.add_argument("--show_imm",   action="store_true",
                        help="Зберегти графік IMM ймовірностей")
    args = parser.parse_args()
    run(args)
