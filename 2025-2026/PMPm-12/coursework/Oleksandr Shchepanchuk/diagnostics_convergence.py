#!/usr/bin/env python3
"""Convergence study for the reaction-diffusion FEM grid."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from surrogatelab.problems import ReactionDiffusionProblem


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "tables"


def main() -> None:
    """Evaluate base QoIs and random-parameter J1 convergence."""
    OUT.mkdir(parents=True, exist_ok=True)
    base_table = _base_convergence()
    j1_table = _random_j1_convergence()

    base_table.to_csv(OUT / "convergence_study.csv", index=False)
    j1_table.to_csv(OUT / "convergence_j1_random.csv", index=False)

    print("Base parameter convergence")
    print(base_table.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print()
    print("J1 random-parameter sanity check")
    print(j1_table.to_string(index=False, float_format=lambda value: f"{value:.6g}"))

    max_rel_41_81 = float(j1_table["rel_41_81"].max())
    if max_rel_41_81 < 0.01:
        print("\nRecommendation: pde_nx=41 is adequate for J1 on this sample.")
    else:
        print(
            "\nRecommendation: use pde_nx=81 for production J1 studies; "
            f"max rel(41,81) = {max_rel_41_81:.3%}."
        )


def _base_convergence() -> pd.DataFrame:
    rows = []
    previous = None
    for n_nodes in [21, 41, 81, 161]:
        problem = ReactionDiffusionProblem(pde_nx=n_nodes, mass="consistent")
        base_mu = problem.p_base[list(problem.param_indices)]
        started = time.perf_counter()
        qoi = problem.evaluate(base_mu)
        elapsed = time.perf_counter() - started

        rows.append(
            {
                "N": n_nodes,
                "J": qoi["J"],
                "J1": qoi["J1"],
                "J2": qoi["J2"],
                "rel_change_J1": _relative_change(qoi["J1"], previous["J1"])
                if previous
                else float("nan"),
                "rel_change_J2": _relative_change(qoi["J2"], previous["J2"])
                if previous
                else float("nan"),
                "time_seconds": elapsed,
            }
        )
        previous = qoi
    return pd.DataFrame(rows)


def _random_j1_convergence() -> pd.DataFrame:
    rng = np.random.default_rng(20240520)
    reference_problem = ReactionDiffusionProblem(pde_nx=21, mass="consistent")
    unit_points = rng.uniform(0.0, 1.0, size=(5, reference_problem.dim))
    parameters = reference_problem.from_unit(unit_points)
    rows = []

    for point_index, mu in enumerate(parameters):
        values = {}
        times = {}
        for n_nodes in [21, 41, 81, 161]:
            problem = ReactionDiffusionProblem(pde_nx=n_nodes, mass="consistent")
            started = time.perf_counter()
            values[n_nodes] = problem.evaluate(mu)["J1"]
            times[n_nodes] = time.perf_counter() - started
        rows.append(
            {
                "point": point_index,
                "J1_21": values[21],
                "J1_41": values[41],
                "J1_81": values[81],
                "J1_161": values[161],
                "rel_21_41": _relative_change(values[41], values[21]),
                "rel_41_81": _relative_change(values[81], values[41]),
                "rel_81_161": _relative_change(values[161], values[81]),
                "converging": _relative_change(values[81], values[41])
                < _relative_change(values[41], values[21]),
                "time_161_seconds": times[161],
            }
        )
    return pd.DataFrame(rows)


def _relative_change(new: float, old: float) -> float:
    return abs(new - old) / max(abs(old), 1e-300)


if __name__ == "__main__":
    main()
