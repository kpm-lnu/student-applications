import re
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_txt(path):

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    frames, cv_x, cv_y, mine_x, mine_y = [], [], [], [], []

    for line in lines:
        # Рядки з даними починаються з |  число  |
        m = re.match(
            r'\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|',
            line.strip()
        )
        if m:
            frames.append(int(m.group(1)))
            cv_x.append(float(m.group(2)))
            cv_y.append(float(m.group(3)))
            mine_x.append(float(m.group(4)))
            mine_y.append(float(m.group(5)))

    # Розміри відео — два числа в кінці файлу
    numbers_at_end = []
    for line in reversed(lines):
        line = line.strip()
        if re.fullmatch(r'\d+', line):
            numbers_at_end.append(int(line))
        if len(numbers_at_end) == 2:
            break

    # Перше знайдене — висота, друге — ширина (бо читали з кінця)
    if len(numbers_at_end) == 2:
        H, W = numbers_at_end[1], numbers_at_end[0]
    else:
        # Якщо не знайдено — беремо з максимальних координат
        W = int(max(max(cv_x), max(mine_x))) + 50
        H = int(max(max(cv_y), max(mine_y))) + 50
        print("[!] Розміри відео не знайдено, використовуємо приблизні")

    return W, H, frames, cv_x, cv_y, mine_x, mine_y


def plot_tracks(txt_path):

    W, H, frames, cv_x, cv_y, mine_x, mine_y = parse_txt(txt_path)
    stem = Path(txt_path).stem

    # ── Графік 1: траєкторія ───────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)

    ax1.plot(cv_x,   cv_y,   color="blue", linewidth=1.5, label="OpenCV")
    ax1.plot(mine_x, mine_y, color="red",  linewidth=1.5, label="Custom")

    ax1.scatter([cv_x[0]],    [cv_y[0]],    color="blue", s=60, zorder=5)
    ax1.scatter([cv_x[-1]],   [cv_y[-1]],   color="blue", s=60, marker="x", zorder=5)
    ax1.scatter([mine_x[0]],  [mine_y[0]],  color="red",  s=60, zorder=5)
    ax1.scatter([mine_x[-1]], [mine_y[-1]], color="red",  s=60, marker="x", zorder=5)

    ax1.set_title("Траєкторія відстежуваної точки", fontsize=13)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()
    ax1.grid(color="#ddd", linewidth=0.5)
    ax1.text(0.01, 0.01, f"Відео: {W}×{H} px",
             transform=ax1.transAxes, color="#888", fontsize=8, va="bottom")

    plt.tight_layout()
    out1 = str(Path(txt_path).with_name(stem + "_trajectory.png"))
    fig1.savefig(out1, dpi=150)
    print(f"Збережено: {out1}")

    plt.show()


if __name__ == "__main__":

    txt_path = Path("processed/frisbee_table_combined.txt")
    plot_tracks(txt_path)