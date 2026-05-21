#!/usr/bin/env python3
"""Build design_panel PDFs for 2D problems.

Each PDF shows the QoI contour in normalised unit-cube parameter space
plus the n_target=40 points that a given sampler places. One PDF per
(problem, sampler) combination — visualises how each strategy
distributes its budget across the domain.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-surrogatelab")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from surrogatelab.problems import (
    AdvDiffAMICon_FD,
    BraninProblem,
    HeatAMICon_FD,
    ReactionDiffusionProblem,
)
from surrogatelab.sampling import get_sampler

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# RD-2D uses lighter mesh for the QoI background only; samplers still
# place points that are then evaluated via the cached forward.
PROBLEMS_2D = [
    ("branin", "f", lambda: BraninProblem(), False),
    ("heat_amicon", "q", lambda: HeatAMICon_FD(n_grid=51), True),
    ("advdiff_amicon", "q", lambda: AdvDiffAMICon_FD(n_grid=51), True),
    ("rd_2d", "J", lambda: ReactionDiffusionProblem(pde_nx=41), True),
]

SAMPLERS = ["LHS", "Halton", "P-greedy", "β-greedy(β=0.5)", "MEPE", "EIGF"]
N_TARGET = 40
N_PER_DIM_BG = 40


def safe_name(sampler: str) -> str:
    return (
        sampler.replace("β-greedy(β=0.5)", "beta_greedy_0.5")
        .replace("β", "beta")
        .replace("=", "")
        .replace("(", "_")
        .replace(")", "")
        .replace("/", "_")
    )


def build_qoi_background(problem, qoi: str, log_scale: bool):
    """Evaluate QoI on a unit-cube grid; returns G1, G2, field (for contour)."""
    u = np.linspace(0.01, 0.99, N_PER_DIM_BG)
    G1, G2 = np.meshgrid(u, u, indexing="xy")
    X_unit = np.column_stack([G1.ravel(), G2.ravel()])
    forward = problem.forward(qoi)
    Z = forward(X_unit)
    field = np.log10(np.abs(Z) + 1e-12) if log_scale else Z
    return G1, G2, field.reshape(G1.shape), log_scale


def plot_panel(slug: str, qoi: str, problem, sampler_name: str,
               G1, G2, field, log_scale: bool):
    sampler = get_sampler(sampler_name)
    if sampler.is_adaptive:
        sampler.min_distance = 0.01
        sampler.scoring_log_transform = log_scale
        forward = problem.forward(qoi)
        design = sampler.build(N_TARGET, problem.dim, seed=0, forward=forward)
    else:
        design = sampler.build(N_TARGET, problem.dim, seed=0)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    cf = ax.contourf(G1, G2, field, levels=18, cmap="cividis", alpha=0.9)
    ax.scatter(
        design[:, 0], design[:, 1],
        c="white", edgecolors="black", s=44, linewidth=0.8, zorder=5,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\mu_1$ (нормовано)")
    ax.set_ylabel(r"$\mu_2$ (нормовано)")
    cb_label = rf"$\log_{{10}}|{qoi}(\mu)|$" if log_scale else rf"${qoi}(\mu)$"
    plt.colorbar(cf, ax=ax, label=cb_label, shrink=0.85)
    ax.set_title(f"{sampler_name}: {N_TARGET} точок на задачі {slug}")
    plt.tight_layout()
    out = FIG / f"design_panel_{slug}_{qoi}_{safe_name(sampler_name)}.pdf"
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    total = len(PROBLEMS_2D) * len(SAMPLERS)
    print(f"Building {total} design_panel PDFs ({len(PROBLEMS_2D)} problems × {len(SAMPLERS)} samplers)")
    print(f"N_TARGET={N_TARGET}, background grid {N_PER_DIM_BG}×{N_PER_DIM_BG}")

    n_done = 0
    n_failed = 0
    for slug, qoi, prob_factory, log_scale in PROBLEMS_2D:
        t0 = time.perf_counter()
        problem = prob_factory()
        print(f"\n[{slug}] building QoI background ({N_PER_DIM_BG}^2 = {N_PER_DIM_BG**2} evals)...")
        G1, G2, field, ls = build_qoi_background(problem, qoi, log_scale)
        print(f"  background ready in {time.perf_counter() - t0:.1f}s")

        for sampler_name in SAMPLERS:
            t1 = time.perf_counter()
            try:
                out = plot_panel(slug, qoi, problem, sampler_name, G1, G2, field, ls)
                dt = time.perf_counter() - t1
                print(f"  OK {out.name}  ({dt:.1f}s)")
                n_done += 1
            except Exception as exc:
                print(f"  FAIL {slug}/{sampler_name}: {exc}")
                n_failed += 1

    print(f"\nDone: {n_done}/{total} PDFs ({n_failed} failed) in {FIG}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
