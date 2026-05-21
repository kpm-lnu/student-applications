#!/usr/bin/env python3
"""Paper-compatible re-evaluation of Heat-AMICon and AdvDiff-AMICon.

Reproduces AMICon-2026 paper methodology:
- Test set: 2000 uniform-random points (seed=0), fixed
- 10 seeds
- n_grid: [5, 10, 15, 20, 30, 40, 60, 80, 100, 120] (paper exact)

Original 81-point full-factorial grid results remain in
`outputs/tables/{slug}_surrogate_*.csv` for §5.3 reproducibility
discussion; paper-compat outputs use suffix `_paper_`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-surrogatelab")

import numpy as np
import pandas as pd

from surrogatelab import (
    AdvDiffAMICon_FD,
    ExperimentConfig,
    HeatAMICon_FD,
    samples_to_tolerance,
    summarise,
)
from surrogatelab import plotting as plot
from surrogatelab.experiment import _fit_and_score
from surrogatelab.sampling import get_sampler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"

# Paper exact settings
N_VAL = 2000
VAL_SEED = 0
N_SEEDS = 10
N_GRID = [5, 10, 15, 20, 30, 40, 60, 80, 100, 120]
TOLERANCES = [0.2, 0.1, 0.05]

SAMPLERS = [
    "Random", "LHS", "Halton", "P-greedy", "f-greedy",
    "β-greedy(β=0.5)", "MEPE", "EIGF",
]

PROBLEMS = [
    ("Heat-AMICon", "heat_amicon", lambda: HeatAMICon_FD(n_grid=51)),
    ("AdvDiff-AMICon", "advdiff_amicon", lambda: AdvDiffAMICon_FD(n_grid=51)),
]


def make_paper_test_set(problem, qoi):
    """2000 uniform-random points in [0,1]^d, seed=0 (paper exact)."""
    rng = np.random.default_rng(VAL_SEED)
    X_test = rng.uniform(0.0, 1.0, size=(N_VAL, problem.dim))
    forward = problem.forward(qoi)
    y_test = forward(X_test)
    return X_test, y_test


def run_one(label, slug, factory):
    print(f"\n=== {label} (paper-compat: {N_VAL} test pts, {N_SEEDS} seeds) ===")
    problem = factory()
    qoi = problem.qoi_names[0]

    X_test, y_test = make_paper_test_set(problem, qoi)
    print(f"  test set: {X_test.shape[0]} random points in [0,1]^{problem.dim}")

    cfg = ExperimentConfig(
        n_grid=N_GRID,
        seeds=list(range(N_SEEDS)),
        surrogate_kernel="gaussian",
        nugget=1e-10,
        log_transform=True,
        min_distance=0.01,
        test_per_dim=11,  # unused; we built test set manually
    )

    rows = []
    t0 = time.perf_counter()

    forward = problem.forward(qoi)
    n_max = max(N_GRID)
    for sampler_name in SAMPLERS:
        sampler = get_sampler(sampler_name)
        if sampler.is_adaptive:
            sampler.min_distance = cfg.min_distance
            sampler.scoring_log_transform = cfg.log_transform
        for seed in cfg.seeds:
            if sampler.is_adaptive:
                full_design = sampler.build(
                    n_max, problem.dim, seed, forward=forward,
                )
                full_values = forward(full_design)
                for n in N_GRID:
                    design = full_design[:n]
                    values = full_values[:n]
                    rec = _fit_and_score(design, values, X_test, y_test, cfg)
                    rec.update(
                        problem=label, qoi=qoi, sampler=sampler_name,
                        seed=seed, n=n,
                    )
                    rows.append(rec)
            else:
                for n in N_GRID:
                    design = sampler.build(n, problem.dim, seed)
                    values = forward(design)
                    rec = _fit_and_score(design, values, X_test, y_test, cfg)
                    rec.update(
                        problem=label, qoi=qoi, sampler=sampler_name,
                        seed=seed, n=n,
                    )
                    rows.append(rec)

    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    df.to_csv(TAB / f"{slug}_paper_results.csv", index=False)

    summary = summarise(df, "nrmse")
    summary.to_csv(TAB / f"{slug}_paper_summary_nrmse.csv", index=False)

    plot.plot_convergence(
        summary, qoi, "nrmse",
        str(FIG / f"convergence_{slug}_q_paper.pdf"),
        ylabel="NRMSE",
    )

    all_tol_rows = []
    for tol in TOLERANCES:
        tol_table = samples_to_tolerance(df, tol)
        tol_table["tolerance"] = tol
        all_tol_rows.append(tol_table)
    tol_combined = pd.concat(all_tol_rows, ignore_index=True)
    tol_combined.to_csv(TAB / f"{slug}_paper_samples_to_tol.csv", index=False)

    tol_for_plot = samples_to_tolerance(df, 0.10)
    plot.plot_samples_to_tol(
        tol_for_plot, qoi, 0.10, n_max,
        str(FIG / f"samples_to_tol_{slug}_q_paper.pdf"),
    )

    print(f"\n  Samples-to-tol для τ=0.10 (n_max={n_max}):")
    for samp in SAMPLERS:
        sub = tol_for_plot[tol_for_plot["sampler"] == samp]
        if len(sub):
            n_star = sub["n_star_median"].iloc[0]
            n_cens = (
                int(sub["n_censored"].iloc[0])
                if "n_censored" in sub.columns else 0
            )
            if pd.isna(n_star):
                status = "✗ censored"
                n_disp = "—"
            else:
                status = "✓" if n_star < n_max else "(at n_max)"
                n_disp = f"{n_star:.0f}"
            print(f"    {samp:25s} n*={n_disp} (cens {n_cens}/{N_SEEDS}) {status}")

    return df, summary


def main() -> int:
    print("=" * 60)
    print("Paper-compatible re-evaluation (AMICon-2026 methodology)")
    print("=" * 60)
    print(f"Test set: {N_VAL} uniform-random points, seed={VAL_SEED}")
    print(f"Seeds: {N_SEEDS}")
    print(f"n-grid: {N_GRID}")

    for label, slug, factory in PROBLEMS:
        run_one(label, slug, factory)

    print("\n" + "=" * 60)
    print("Done. Original 81-grid CSVs preserved as is.")
    print("Paper-compat CSVs have suffix '_paper_'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
