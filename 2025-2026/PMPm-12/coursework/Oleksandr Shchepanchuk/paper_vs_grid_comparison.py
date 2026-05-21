#!/usr/bin/env python3
"""Side-by-side: 81-point grid vs paper 2000-random test set."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
TAB = OUT / "tables"


def main() -> int:
    print("=" * 80)
    print("PAPER vs 81-GRID comparison (Heat-AMICon and AdvDiff-AMICon)")
    print("=" * 80)

    for slug, label in [
        ("heat_amicon", "Heat-AMICon"),
        ("advdiff_amicon", "AdvDiff-AMICon"),
    ]:
        print(f"\n=== {label} ===")

        grid_path = TAB / f"{slug}_surrogate_results.csv"
        paper_path = TAB / f"{slug}_paper_results.csv"
        if not grid_path.exists() or not paper_path.exists():
            print(f"  MISS: {grid_path.name} or {paper_path.name}")
            continue

        df_grid = pd.read_csv(grid_path)
        df_paper = pd.read_csv(paper_path)

        grid_at_120 = df_grid[df_grid["n"] == 120].groupby("sampler")["nrmse"].median()
        paper_at_120 = df_paper[df_paper["n"] == 120].groupby("sampler")["nrmse"].median()

        print(f"{'Sampler':25s} {'81-grid':>12s} {'2000-rand':>12s} {'ratio':>10s}")
        print("-" * 65)
        for samp in grid_at_120.index:
            g = float(grid_at_120.get(samp, float("nan")))
            p = float(paper_at_120.get(samp, float("nan")))
            ratio = g / p if p > 0 else float("nan")
            print(f"{samp:25s} {g:12.4f} {p:12.4f} {ratio:10.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
