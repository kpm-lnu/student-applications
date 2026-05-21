"""
ablation.py
Автоматичне ablation study: порівняння всіх конфігурацій
від базового B1 до повної запропонованої системи P.

Конфігурації (з розд. 2, табл. 2.4):
  B1 — YOLOv8s          + ByteTrack CV-KF   (повний baseline)
  B2 — YOLOv8s + P2     + ByteTrack CV-KF
  B3 — YOLOv8s + P2 + SimAM + ByteTrack CV-KF
  B4 — YOLOv8s + P2 + SimAM + WiseIoU + ByteTrack CV-KF
  P  — YOLOv8s + P2 + SimAM + WiseIoU + ByteTrack IMM-KF

Запуск:
    python ablation.py --weights_b1  checkpoints/b1.pt  \
                       --weights_b2  checkpoints/b2.pt  \
                       --weights_b3  checkpoints/b3.pt  \
                       --weights_b4  checkpoints/b4.pt  \
                       --weights_p   checkpoints/best_final.pt

Або якщо є лише фінальні ваги P — порівнюємо лише трекери:
    python ablation.py --weights_p checkpoints/best_final.pt \
                       --tracker_only
"""

import argparse
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # без GUI
import matplotlib.pyplot as plt
from tabulate import tabulate   # pip install tabulate

from config        import DATASET_ROOT, RESULTS_DIR, CONF_THRESH
from detector      import Detector
from tracker       import ByteTrackIMM
from tracker_baseline import ByteTrackCV
from dataset_utils import load_tracking_split


# ─────────────────────────────────────────────────────────────
#  Опис конфігурацій
# ─────────────────────────────────────────────────────────────

CONFIGS = {
    "B1": dict(label="YOLOv8s + CV-KF",
               tracker="cv",  weights_key="weights_b1"),
    "B2": dict(label="YOLOv8s-P2 + CV-KF",
               tracker="cv",  weights_key="weights_b2"),
    "B3": dict(label="YOLOv8s-P2-SimAM + CV-KF",
               tracker="cv",  weights_key="weights_b3"),
    "B4": dict(label="YOLOv8s-P2-SimAM-WIoU + CV-KF",
               tracker="cv",  weights_key="weights_b4"),
    "P":  dict(label="YOLOv8s-P2-SimAM-WIoU + IMM-KF",
               tracker="imm", weights_key="weights_p"),
}


# ─────────────────────────────────────────────────────────────
#  Оцінка однієї конфігурації
# ─────────────────────────────────────────────────────────────

def eval_config(cfg_name: str,
                cfg:      dict,
                weights:  str,
                sequences: list) -> dict:
    """
    Запускає детектор + трекер відповідної конфігурації
    на всіх test-послідовностях.
    Повертає агреговані метрики.
    """
    print(f"\n[Ablation] {cfg_name}: {cfg['label']}")
    print(f"           Ваги: {weights}")

    detector = Detector(weights, conf=CONF_THRESH)
    detector.warmup()

    tracker_cls = ByteTrackIMM if cfg["tracker"] == "imm" \
                  else ByteTrackCV

    all_tp = all_fp = all_fn = all_idsw = 0
    iou_sum = 0.0; iou_count = 0
    fps_list = []
    mu_hist_all = {}

    for seq in sequences:
        tracker = tracker_cls()
        gts_c   = seq.gt_to_center()

        tp = fp = fn = idsw = 0
        prev_id: Optional[int] = None
        mu_buf = []

        for frame_idx, (frame, _) in enumerate(seq):
            gt     = gts_c[frame_idx]
            gt_vis = (gt[2] > 0 and gt[3] > 0)

            t0     = time.perf_counter()
            dets   = detector.detect(frame)
            tracks = tracker.update(dets)
            fps_list.append(1.0 / max(time.perf_counter() - t0, 1e-9))

            # Запис IMM ймовірностей (лише для конфігурації P)
            if cfg["tracker"] == "imm" and tracks:
                mu_buf.append(tracks[0].model_probs.copy())

            if not gt_vis:
                fp += len(tracks); continue

            if tracks:
                tbboxes = np.array([t.bbox for t in tracks])
                ious    = _iou_one_gt(gt, tbboxes)
                bi      = int(np.argmax(ious))
                best    = ious[bi]

                if best >= 0.5:
                    tp     += 1
                    iou_sum += best; iou_count += 1
                    fp     += len(tracks) - 1
                    cur_id  = tracks[bi].id
                    if prev_id is not None and prev_id != cur_id:
                        idsw += 1
                    prev_id = cur_id
                else:
                    fn += 1; fp += len(tracks)
            else:
                fn += 1

        all_tp   += tp;  all_fp += fp
        all_fn   += fn;  all_idsw += idsw

        if mu_buf:
            mu_hist_all[seq.name] = np.array(mu_buf)

    gt_total = all_tp + all_fn
    mota = 1.0 - (all_fn + all_fp + all_idsw) / max(gt_total, 1)
    motp = iou_sum / max(iou_count, 1)
    idf1 = 2*all_tp / max(2*all_tp + all_fp + all_fn, 1)
    deta = all_tp / max(all_tp + all_fp + all_fn, 1)
    assa = all_tp / max(all_tp + all_idsw, 1)
    hota = float(np.sqrt(max(deta * assa, 0)))
    fps  = float(np.mean(fps_list[10:])) if len(fps_list) > 10 else 0.0

    return {
        "config":  cfg_name,
        "label":   cfg["label"],
        "MOTA":    round(mota, 4),
        "MOTP":    round(motp, 4),
        "IDF1":    round(idf1, 4),
        "HOTA":    round(hota, 4),
        "IDSW":    all_idsw,
        "FPS":     round(fps, 1),
        "_mu_hist": mu_hist_all,
    }


def _iou_one_gt(gt, tracks):
    def xyxy(b):
        return np.array([b[0]-b[2]/2, b[1]-b[3]/2,
                         b[0]+b[2]/2, b[1]+b[3]/2])
    g = xyxy(gt); ag = gt[2]*gt[3]
    ious = []
    for t in tracks:
        tx = xyxy(t)
        ix1,iy1 = max(g[0],tx[0]), max(g[1],tx[1])
        ix2,iy2 = min(g[2],tx[2]), min(g[3],tx[3])
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        union = ag + t[2]*t[3] - inter
        ious.append(inter/max(union,1e-6))
    return np.array(ious)


# ─────────────────────────────────────────────────────────────
#  Виведення та збереження результатів
# ─────────────────────────────────────────────────────────────

METRICS = ["MOTA", "MOTP", "IDF1", "HOTA", "IDSW", "FPS"]

def print_table(results: list) -> None:
    headers = ["Конфіг", "Назва"] + METRICS
    rows = []
    for r in results:
        row = [r["config"], r["label"]] + [r[m] for m in METRICS]
        rows.append(row)
    print("\n" + "="*70)
    print("ABLATION STUDY — Результати трекінгу")
    print("="*70)
    print(tabulate(rows, headers=headers, tablefmt="github",
                   floatfmt=".4f"))


def save_results(results: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    clean = [{k: v for k, v in r.items() if k != "_mu_hist"}
             for r in results]
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(clean, f, indent=2)

    # CSV для LaTeX
    with open(out_dir / "ablation_results.csv", "w") as f:
        f.write("Config,Label," + ",".join(METRICS) + "\n")
        for r in clean:
            line = f"{r['config']},{r['label']}," + \
                   ",".join(str(r[m]) for m in METRICS)
            f.write(line + "\n")

    print(f"\n[Ablation] Результати збережено: {out_dir}")


def plot_comparison(results: list, out_dir: Path) -> None:
    """Bar chart для MOTA, IDF1, HOTA по конфігураціях."""
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [r["config"] for r in results]
    colors  = ["#aacbe8", "#72b4d8", "#3a8fbf",
               "#1a5e8a", "#e07b39"]

    for metric in ["MOTA", "IDF1", "HOTA"]:
        vals = [r[metric] for r in results]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(configs, vals, color=colors[:len(configs)],
                      width=0.55, edgecolor="black", linewidth=0.7)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_ylim(0, min(1.0, max(vals)*1.2))
        ax.set_ylabel(metric)
        ax.set_title(f"Ablation Study — {metric}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"ablation_{metric.lower()}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Ablation] Графік: {path}")


def plot_idsw_comparison(results: list, out_dir: Path) -> None:
    """Окремий графік IDSW (менше — краще)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = [r["config"] for r in results]
    vals    = [r["IDSW"] for r in results]
    colors  = ["#aacbe8","#72b4d8","#3a8fbf","#1a5e8a","#e07b39"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(configs, vals, color=colors[:len(configs)],
                  width=0.55, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_ylabel("IDSW (↓ краще)")
    ax.set_title("Ablation Study — кількість змін ID (IDSW)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "ablation_idsw.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_imm_probs(results: list, out_dir: Path) -> None:
    """
    Графік ймовірностей IMM для конфігурації P
    по першій доступній послідовності.
    """
    for r in results:
        if r["config"] == "P" and r.get("_mu_hist"):
            seq_name = next(iter(r["_mu_hist"]))
            mu_arr   = r["_mu_hist"][seq_name]
            if len(mu_arr) == 0:
                continue

            fig, ax = plt.subplots(figsize=(11, 3))
            for j, (lbl, col) in enumerate(
                    zip(["CV","CA","CT"],
                        ["tab:blue","tab:orange","tab:green"])):
                ax.plot(mu_arr[:,j], label=lbl, color=col, lw=1.5)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Кадр"); ax.set_ylabel("Ймовірність моделі")
            ax.set_title(f"IMM — динаміка ймовірностей: {seq_name}")
            ax.legend(loc="upper right"); ax.grid(alpha=0.3)
            fig.tight_layout()
            path = out_dir / f"imm_probs_{seq_name}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[Ablation] IMM-графік: {path}")
            break


# ─────────────────────────────────────────────────────────────
#  Режим tracker_only: порівнюємо лише трекери
# ─────────────────────────────────────────────────────────────

def tracker_only_comparison(weights_p: str, out_dir: Path) -> None:
    """
    Запускає ті самі ваги з двома трекерами: CV-KF і IMM-KF.
    Корисно якщо є лише фінальні ваги.
    """
    print("\n[Ablation] Режим tracker_only: CV-KF vs IMM-KF")
    sequences = load_tracking_split(
        DATASET_ROOT / "tracking" / "test"
    )

    configs_slim = {
        "B_tracker": dict(label="+ CV-KF (базовий трекер)",
                          tracker="cv",  weights_key="dummy"),
        "P_tracker": dict(label="+ IMM-KF (запропонований)",
                          tracker="imm", weights_key="dummy"),
    }

    results = []
    for name, cfg in configs_slim.items():
        r = eval_config(name, cfg, weights_p, sequences)
        results.append(r)

    print_table(results)
    save_results(results, out_dir)
    plot_comparison(results, out_dir / "plots")
    plot_idsw_comparison(results, out_dir / "plots")
    plot_imm_probs(results, out_dir / "plots")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main(args):
    out_dir   = RESULTS_DIR / "ablation"
    sequences = load_tracking_split(
        DATASET_ROOT / "tracking" / "test"
    )

    if args.tracker_only:
        tracker_only_comparison(args.weights_p, out_dir)
        return

    # Повне ablation study
    weights_map = {
        "weights_b1": args.weights_b1,
        "weights_b2": args.weights_b2,
        "weights_b3": args.weights_b3,
        "weights_b4": args.weights_b4,
        "weights_p":  args.weights_p,
    }

    results = []
    for cfg_name, cfg in CONFIGS.items():
        wk = cfg["weights_key"]
        w  = weights_map.get(wk)
        if not w or not Path(w).exists():
            print(f"[Ablation] Пропускаємо {cfg_name}: "
                  f"ваги не знайдено ({w})")
            continue
        r = eval_config(cfg_name, cfg, w, sequences)
        results.append(r)

    if not results:
        print("[Ablation] Жодної конфігурації не оцінено.")
        return

    print_table(results)
    save_results(results, out_dir)
    plot_comparison(results, out_dir / "plots")
    plot_idsw_comparison(results, out_dir / "plots")
    plot_imm_probs(results, out_dir / "plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation study: B1 → B2 → B3 → B4 → P"
    )
    parser.add_argument("--weights_b1", type=str, default="")
    parser.add_argument("--weights_b2", type=str, default="")
    parser.add_argument("--weights_b3", type=str, default="")
    parser.add_argument("--weights_b4", type=str, default="")
    parser.add_argument("--weights_p",  type=str,
                        default="checkpoints/best_final.pt")
    parser.add_argument("--tracker_only", action="store_true",
                        help="Порівняти лише CV-KF vs IMM-KF "
                             "на одних ваги (weights_p)")
    args = parser.parse_args()
    main(args)
