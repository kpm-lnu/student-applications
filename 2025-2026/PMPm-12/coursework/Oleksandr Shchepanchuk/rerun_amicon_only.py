#!/usr/bin/env python3
"""Re-run ONLY Heat-AMICon and AdvDiff-AMICon with extended n_max=120.

Production-run with n_max=80 was insufficient - no sampler reached
tolerance tau=0.1. This script reruns these two problems only (others
stay as is) and updates summary.json incrementally.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-surrogatelab")

import pandas as pd

from surrogatelab import (
    AdvDiffAMICon_FD,
    ExperimentConfig,
    HeatAMICon_FD,
    run_comparison,
    samples_to_tolerance,
    summarise,
)
from surrogatelab import plotting as plot

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"

SAMPLERS = [
    "Random", "LHS", "Halton", "P-greedy", "f-greedy",
    "β-greedy(β=0.5)", "MEPE", "EIGF",
]

PROBLEMS = [
    ("Heat-AMICon", lambda: HeatAMICon_FD(n_grid=51), 0.10),
    ("AdvDiff-AMICon", lambda: AdvDiffAMICon_FD(n_grid=51), 0.10),
]

BUDGETS = list(range(5, 121, 5))  # 5, 10, 15, ..., 120


def slug(label: str) -> str:
    return label.lower().replace("-", "_")


def run_one(label, factory, tolerance):
    problem = factory()
    qois = problem.qoi_names
    print(f"\n=== {label}: dim={problem.dim}, qois={qois}, n_max={BUDGETS[-1]} ===")
    cfg = ExperimentConfig(
        n_grid=BUDGETS,
        seeds=[0],
        surrogate_kernel="gaussian",
        nugget=1e-10,
        log_transform=True,
        min_distance=0.01,
        test_per_dim=9,
    )
    t0 = time.perf_counter()
    results, _designs = run_comparison(
        problem, qois, SAMPLERS, cfg, verbose=False,
    )
    elapsed = time.perf_counter() - t0
    results.insert(0, "problem", label)

    s = slug(label)
    results.to_csv(TAB / f"{s}_surrogate_results.csv", index=False)
    summary = summarise(results, "nrmse")
    summary.to_csv(TAB / f"{s}_surrogate_summary_nrmse.csv", index=False)

    tol_rows_all = []
    for qoi in qois:
        plot.plot_convergence(
            summary, qoi, "nrmse",
            str(FIG / f"convergence_{s}_{qoi}.pdf"),
            ylabel="NRMSE",
        )
        tol_rows = samples_to_tolerance(
            results[results["qoi"] == qoi], tolerance,
        )
        tol_rows_all.append(tol_rows)
        plot.plot_samples_to_tol(
            tol_rows, qoi, tolerance, BUDGETS[-1],
            str(FIG / f"samples_to_tol_{s}_{qoi}.pdf"),
        )
    tol_table = pd.concat(tol_rows_all, ignore_index=True)
    tol_table.to_csv(TAB / f"{s}_samples_to_tolerance.csv", index=False)

    n_max = BUDGETS[-1]
    nrmse_at_max = {}
    for qoi in qois:
        nrmse_at_max[qoi] = {}
        for samp in SAMPLERS:
            sel = summary[
                (summary["qoi"] == qoi)
                & (summary["sampler"] == samp)
                & (summary["n"] == n_max)
            ]["median"]
            nrmse_at_max[qoi][samp] = float(sel.iloc[0]) if len(sel) else float("nan")

    samples_to_tol_map = {}
    for qoi in qois:
        sub = tol_table[tol_table["qoi"] == qoi].set_index("sampler")
        samples_to_tol_map[qoi] = {}
        for samp in SAMPLERS:
            if samp in sub.index:
                v = sub.loc[samp, "n_star_median"]
                samples_to_tol_map[qoi][samp] = None if pd.isna(v) else float(v)
            else:
                samples_to_tol_map[qoi][samp] = None

    print(f"  done in {elapsed:.1f}s")
    print(f"  median NRMSE@{n_max} for {qois[0]}:")
    print("    " + "  ".join(
        f"{sn}={nrmse_at_max[qois[0]][sn]:.3g}" for sn in SAMPLERS
    ))

    return {
        "qois": list(qois),
        "tolerance": tolerance,
        "n_grid": BUDGETS,
        "n_solver_calls": int(getattr(problem, "n_solver_calls", 0)),
        "median_nrmse_at_max_budget": nrmse_at_max,
        "samples_to_tolerance": samples_to_tol_map,
    }


def main() -> int:
    print("Re-running Heat-AMICon and AdvDiff-AMICon with n_max=120")
    new_summaries = {}
    total_t = 0.0
    total_solves = 0
    for label, factory, tol in PROBLEMS:
        t0 = time.perf_counter()
        summary = run_one(label, factory, tol)
        total_t += time.perf_counter() - t0
        total_solves += summary["n_solver_calls"]
        new_summaries[label] = summary

    summary_path = OUT / "summary.json"
    with summary_path.open() as f:
        s = json.load(f)

    for label, summary in new_summaries.items():
        s["median_nrmse_at_max_budget"][label] = summary["median_nrmse_at_max_budget"]
        s["samples_to_tolerance"][label] = summary["samples_to_tolerance"]
        speedup = {}
        for qoi in summary["qois"]:
            speedup[qoi] = {}
            lhs_n = summary["samples_to_tolerance"][qoi].get("LHS")
            for samp, n in summary["samples_to_tolerance"][qoi].items():
                if lhs_n is not None and n is not None and n > 0:
                    speedup[qoi][samp] = lhs_n / n
                else:
                    speedup[qoi][samp] = None
        s["speedup_vs_LHS"][label] = speedup
        if "qois" in s and label in s["qois"]:
            s["qois"][label] = summary["qois"]

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    plot.plot_overall_summary(s, str(FIG / "overall_summary.pdf"))

    print(f"\n=== Готово ===")
    print(f"Загальний час: {total_t:.1f}s")
    print(f"FOM solves (Heat+AdvDiff): {total_solves}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
