#!/usr/bin/env python3
"""Regenerate Part-A illustration PNGs (RD model physics) with the
current (Ukrainian) plotting labels.

These were originally produced before the label translation, so the
files on disk still showed English titles. Run this whenever the
captions in `surrogatelab.plotting.plot_ode_solution`,
`plot_phase_portrait`, `plot_pde_surface`, `plot_pde_traces` or
`plot_spatial_phase` change.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-surrogatelab")

import numpy as np

from surrogatelab import plotting as plot
from surrogatelab.problems import ReactionDiffusionProblem

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> int:
    problem = ReactionDiffusionProblem(pde_nx=21)

    print("Solving ODE...")
    t0 = time.perf_counter()
    t_ode, Y_ode = problem.solve_ode()
    print(f"  done in {time.perf_counter() - t0:.1f}s; Y shape={Y_ode.shape}")

    plot.plot_ode_solution(t_ode, Y_ode, str(FIG / "ode_solution.png"))
    print("  OK ode_solution.png")
    plot.plot_phase_portrait(problem, t_ode, Y_ode, str(FIG / "ode_phase_portrait.png"))
    print("  OK ode_phase_portrait.png")

    print("\nSolving PDE (this takes a minute)...")
    t0 = time.perf_counter()
    x, t_pde, Y_pde = problem.solve_pde(n_x=161, n_t=301)
    print(f"  done in {time.perf_counter() - t0:.1f}s; Y shape={Y_pde.shape}")

    plot.plot_pde_surface(x, t_pde, Y_pde, which=0, path=str(FIG / "pde_surface_y1.png"))
    print("  OK pde_surface_y1.png")
    plot.plot_pde_surface(x, t_pde, Y_pde, which=1, path=str(FIG / "pde_surface_y2.png"))
    print("  OK pde_surface_y2.png")
    plot.plot_pde_traces(x, t_pde, Y_pde, problem, str(FIG / "pde_traces.png"))
    print("  OK pde_traces.png")

    # Spatial phase: y2'' = -(p4/D2) y2 on the y1=0 branch.
    p4 = problem.p_base[3]
    D2 = problem.D2
    omega = float(np.sqrt(p4 / D2))
    plot.plot_spatial_phase(omega, str(FIG / "spatial_phase.png"))
    print(f"  OK spatial_phase.png (omega={omega:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
