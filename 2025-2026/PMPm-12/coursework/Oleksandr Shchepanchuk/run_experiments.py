#!/usr/bin/env python3
"""Reproduces all numerical experiments for the master's coursework on
adaptive sampling for RBF surrogates.

The production configuration uses ``pde_nx=81`` for reaction-diffusion
models because the J1 convergence sanity check shows that ``pde_nx=41``
is not within 1% of the 81-node result on random parameter samples.

Run ``python run_experiments.py --quick`` for a smoke test over all
configured problems and samplers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-surrogatelab")

from surrogatelab import (
    AdvDiffAMICon_FD,
    BraninProblem,
    ExperimentConfig,
    HeatAMICon_FD,
    Problem,
    ReactionDiffusionProblem,
    run_comparison,
    samples_to_tolerance,
    summarise,
)
from surrogatelab import plotting as plot

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"


PRODUCTION_SAMPLERS = [
    "Random",
    "LHS",
    "Halton",
    "P-greedy",
    "f-greedy",
    "β-greedy(β=0.5)",
    "MEPE",
    "EIGF",
]


def _production_problems(pde_nx: int) -> list[tuple[str, Callable[[], Problem]]]:
    return [
        ("Branin", lambda: BraninProblem()),
        ("Heat-AMICon", lambda: HeatAMICon_FD(n_grid=51)),
        ("AdvDiff-AMICon", lambda: AdvDiffAMICon_FD(n_grid=51)),
        ("RD-2D", lambda: ReactionDiffusionProblem(pde_nx=pde_nx)),
        (
            "RD-4D",
            lambda: ReactionDiffusionProblem.with_4d_params(pde_nx=pde_nx),
        ),
    ]


PRODUCTION_BUDGETS: dict[str, list[int]] = {
    "Branin": list(range(5, 61, 5)),
    "Heat-AMICon": list(range(5, 121, 5)),
    "AdvDiff-AMICon": list(range(5, 121, 5)),
    "RD-2D": list(range(5, 61, 5)),
    "RD-4D": list(range(10, 151, 10)),
}

QUICK_BUDGETS: dict[str, list[int]] = {
    "Branin": [5, 15, 25],
    "Heat-AMICon": [5, 15, 25],
    "AdvDiff-AMICon": [5, 15, 25],
    "RD-2D": [5, 15, 25],
    "RD-4D": [10, 30, 60],
}

# Branin values lie in ~[0.4, 308] without scale-spanning behaviour, so the
# log-transform only adds noise; for the PDE QoIs (which can span orders of
# magnitude) the log fit is the right choice.
LOG_TRANSFORM_PER_PROBLEM: dict[str, bool] = {
    "Branin": False,
    "Heat-AMICon": True,
    "AdvDiff-AMICon": True,
    "RD-2D": True,
    "RD-4D": True,
}


CONFIG = {
    "pde_nx": 81,
    "samplers": PRODUCTION_SAMPLERS,
    "budgets": PRODUCTION_BUDGETS,
    "seeds": list(range(1)),  # 1 seed для швидкості; обґрунтовано у тексті курсової
    "surrogate_kernel": "gaussian",
    "nugget": 1e-10,
    "log_transform": True,
    "min_distance": 0.01,
    "test_per_dim_2d": 9,
    "test_per_dim_4d": 4,
    # τ per problem (NRMSE) — analytic Branin 0.05, AMICon 0.10, RD 0.20.
    "tolerances": {
        "Branin": 0.05,
        "Heat-AMICon": 0.10,
        "AdvDiff-AMICon": 0.10,
        "RD-2D": 0.20,
        "RD-4D": 0.20,
    },
}

QUICK = {
    "pde_nx": 41,
    "budgets": QUICK_BUDGETS,
    "seeds": [0, 1],
    "test_per_dim_2d": 5,
    "test_per_dim_4d": 3,
}


def main() -> int:
    """Run the configured equal-budget experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    args = parser.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    config = dict(CONFIG)
    if args.quick:
        config.update(QUICK)

    problems = _production_problems(config["pde_nx"])
    pde_nx = config["pde_nx"]
    n_seeds = len(config["seeds"])
    budgets = config["budgets"]

    mode = "quick" if args.quick else "production"
    print(f"[run_experiments] mode = {mode}")
    print(f"[run_experiments] pde_nx = {pde_nx}")
    print(f"[run_experiments] n_seeds = {n_seeds}")
    print(f"[run_experiments] samplers = {PRODUCTION_SAMPLERS}")
    print(f"[run_experiments] log_transform_per_problem = {LOG_TRANSFORM_PER_PROBLEM}")
    print("[run_experiments] expected budgets per problem:")
    for label, _ in problems:
        print(f"    {label}: {budgets[label]}")

    if not args.quick:
        assert pde_nx == 81, f"production mode must use pde_nx=81, got {pde_nx}"
        assert n_seeds == 1, f"production mode uses n_seeds=1, got {n_seeds}"
        assert budgets["RD-4D"][-1] == 150, (
            "RD-4D must reach n=150 in production"
        )
        assert budgets["Heat-AMICon"][-1] == 120, (
            "Heat-AMICon must reach n=120 in production"
        )
        assert budgets["AdvDiff-AMICon"][-1] == 120, (
            "AdvDiff-AMICon must reach n=120 in production"
        )

    started = time.perf_counter()
    problem_summaries: dict[str, dict] = {}
    all_results: list[pd.DataFrame] = []

    for label, factory in problems:
        problem = factory()
        qois = problem.qoi_names
        cfg = _experiment_config(problem, label, config)
        print(
            f"\n=== {label}: dim={problem.dim}, qois={qois}, "
            f"n_max={cfg.n_max}, seeds={len(cfg.seeds)} ==="
        )
        summary, results = _run_one_problem(label, problem, qois, cfg, config)
        problem_summaries[label] = summary
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(TAB / "surrogate_results_all_problems.csv", index=False)

    elapsed = time.perf_counter() - started
    summary_payload = _summary_json(problem_summaries, config, elapsed)
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)
    plot.plot_overall_summary(summary_payload, str(FIG / "overall_summary.pdf"))

    total_solves = summary_payload["total_fom_solves"]
    print(
        f"\nDone in {summary_payload['wall_clock_seconds']:.1f}s. "
        f"FOM solves: {total_solves}. Outputs in {OUT}"
    )
    return 0


def _experiment_config(
    problem: Problem,
    label: str,
    config: dict,
) -> ExperimentConfig:
    is_4d = problem.dim == 4
    n_grid = config["budgets"][label]
    log_transform = LOG_TRANSFORM_PER_PROBLEM.get(label, config["log_transform"])
    return ExperimentConfig(
        n_grid=n_grid,
        seeds=config["seeds"],
        surrogate_kernel=config["surrogate_kernel"],
        nugget=config["nugget"],
        log_transform=log_transform,
        min_distance=config["min_distance"],
        test_per_dim=config["test_per_dim_4d"] if is_4d else config["test_per_dim_2d"],
    )


def _run_one_problem(
    label: str,
    problem: Problem,
    qois: list[str],
    cfg: ExperimentConfig,
    config: dict,
) -> tuple[dict, pd.DataFrame]:
    results, _designs = run_comparison(
        problem, qois, config["samplers"], cfg, verbose=False,
    )
    results.insert(0, "problem", label)
    slug = _slug(label)
    results.to_csv(TAB / f"{slug}_surrogate_results.csv", index=False)

    summary = summarise(results, "nrmse")
    summary.to_csv(TAB / f"{slug}_surrogate_summary_nrmse.csv", index=False)
    for qoi in qois:
        plot.plot_convergence(
            summary,
            qoi,
            "nrmse",
            str(FIG / f"convergence_{slug}_{qoi}.pdf"),
            ylabel="NRMSE",
        )

    tolerance = config["tolerances"].get(label, 0.10)
    tolerance_rows = []
    for qoi in qois:
        tolerance_rows.append(
            samples_to_tolerance(results[results["qoi"] == qoi], tolerance)
        )
    tolerance_table = pd.concat(tolerance_rows, ignore_index=True)
    tolerance_table.to_csv(TAB / f"{slug}_samples_to_tolerance.csv", index=False)

    samples_to_tol = {
        qoi: _samples_to_tol_map(tolerance_table, qoi, config["samplers"])
        for qoi in qois
    }
    speedup_vs_lhs = {
        qoi: _speedup_vs_lhs(samples_to_tol[qoi]) for qoi in qois
    }
    nrmse_at_nmax = {
        qoi: _nrmse_at_nmax(summary, qoi, config["samplers"], cfg.n_max)
        for qoi in qois
    }

    primary_qoi = qois[0]
    print(f"  median NRMSE@{cfg.n_max} for {primary_qoi}:")
    line = "  ".join(
        f"{name}={nrmse_at_nmax[primary_qoi][name]:.3g}"
        for name in config["samplers"]
    )
    print("  " + line)

    return (
        {
            "qois": qois,
            "n_grid": cfg.n_grid,
            "seeds": cfg.seeds,
            "tolerance": tolerance,
            "median_nrmse_at_max_budget": nrmse_at_nmax,
            "samples_to_tolerance": samples_to_tol,
            "speedup_vs_LHS": speedup_vs_lhs,
            "n_solver_calls": int(getattr(problem, "n_solver_calls", 0)),
        },
        results,
    )


def _nrmse_at_nmax(
    summary: pd.DataFrame,
    qoi: str,
    samplers: list[str],
    n_max: int,
) -> dict[str, float]:
    values = {}
    for sampler in samplers:
        selected = summary[
            (summary["qoi"] == qoi)
            & (summary["sampler"] == sampler)
            & (summary["n"] == n_max)
        ]["median"]
        values[sampler] = float(selected.iloc[0]) if len(selected) else float("nan")
    return values


def _samples_to_tol_map(
    tolerance_table: pd.DataFrame,
    qoi: str,
    samplers: list[str],
) -> dict[str, float | None]:
    rows = tolerance_table[tolerance_table["qoi"] == qoi].set_index("sampler")
    values: dict[str, float | None] = {}
    for sampler in samplers:
        if sampler not in rows.index:
            values[sampler] = None
            continue
        value = float(rows.loc[sampler, "n_star_median"])
        values[sampler] = None if math.isnan(value) else value
    return values


def _speedup_vs_lhs(samples_to_tol: dict[str, float | None]) -> dict[str, float | None]:
    lhs = samples_to_tol.get("LHS")
    speedups: dict[str, float | None] = {}
    for sampler, value in samples_to_tol.items():
        if lhs is None or value is None or value == 0:
            speedups[sampler] = None
        else:
            speedups[sampler] = lhs / value
    return speedups


def _summary_json(
    problem_summaries: dict,
    config: dict,
    elapsed: float,
) -> dict:
    total_solves = int(
        sum(item["n_solver_calls"] for item in problem_summaries.values())
    )
    return {
        "problems": list(problem_summaries),
        "samplers": config["samplers"],
        "qois": {
            label: summary["qois"]
            for label, summary in problem_summaries.items()
        },
        "tolerances": {
            label: summary["tolerance"]
            for label, summary in problem_summaries.items()
        },
        "median_nrmse_at_max_budget": {
            label: summary["median_nrmse_at_max_budget"]
            for label, summary in problem_summaries.items()
        },
        "samples_to_tolerance": {
            label: summary["samples_to_tolerance"]
            for label, summary in problem_summaries.items()
        },
        "speedup_vs_LHS": {
            label: summary["speedup_vs_LHS"]
            for label, summary in problem_summaries.items()
        },
        "config": {
            "nugget": config["nugget"],
            "kernel": config["surrogate_kernel"],
            "eps_selection": "rippa_loocv",
            "pde_nx_production": CONFIG["pde_nx"],
            "pde_nx": config["pde_nx"],
            "n_seeds": len(config["seeds"]),
            "n_seeds_explanation": "1 seed for time budget; deterministic comparison at fixed initial configuration (vs AMICon-2026 paper which used 10 seeds for IQR estimation)",
            "log_transform": config["log_transform"],
            "log_transform_per_problem": LOG_TRANSFORM_PER_PROBLEM,
            "min_distance": config["min_distance"],
        },
        "wall_clock_seconds": round(elapsed, 1),
        "total_fom_solves": total_solves,
    }


def _slug(label: str) -> str:
    return label.lower().replace("-", "_")


if __name__ == "__main__":
    sys.exit(main())
