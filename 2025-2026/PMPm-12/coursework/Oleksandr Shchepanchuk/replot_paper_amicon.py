#!/usr/bin/env python3
"""Regenerate paper-compat AMICon convergence and samples_to_tol PDFs
from already-saved CSVs (no FOM solves, no Kriging/RBF refits).

Needed because plot_convergence / plot_samples_to_tol labels were
updated after the original paper_eval_amicon.py run.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-surrogatelab")

import pandas as pd

from surrogatelab import plotting as plot
from surrogatelab.experiment import samples_to_tolerance, summarise

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

PROBLEMS = [
    ("heat_amicon", "q", 0.10, 120),
    ("advdiff_amicon", "q", 0.10, 120),
]


def main() -> int:
    for slug, qoi, tol, n_max in PROBLEMS:
        results_path = TAB / f"{slug}_paper_results.csv"
        if not results_path.exists():
            print(f"  skip {slug}: {results_path.name} missing")
            continue
        df = pd.read_csv(results_path)
        summary = summarise(df, "nrmse")

        plot.plot_convergence(
            summary, qoi, "nrmse",
            str(FIG / f"convergence_{slug}_q_paper.pdf"),
            ylabel="NRMSE",
        )
        tol_table = samples_to_tolerance(df, tol)
        plot.plot_samples_to_tol(
            tol_table, qoi, tol, n_max,
            str(FIG / f"samples_to_tol_{slug}_q_paper.pdf"),
        )
        print(f"  OK {slug}: convergence + samples_to_tol paper PDFs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
