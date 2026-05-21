#!/usr/bin/env python3
"""Diagnostic sweep for the stable β-greedy family.

The sweep covers β ∈ {0, 0.25, 0.5, 0.75, 1} with P-greedy and f-greedy as
endpoint sanity checks (β=0 ↔ P-greedy, β=1 ↔ f-greedy)
(Wenzel-Santin-Haasdonk 2023, Constr. Approx. 57:45-74).
"""
from __future__ import annotations

import warnings
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-surrogatelab")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surrogatelab.experiment import make_test_set
from surrogatelab.problems import ReactionDiffusionProblem
from surrogatelab.sampling import (
    BetaGreedySampler,
    FGreedySampler,
    GreedyContext,
    PGreedySampler,
    _membership_mask,
)
from surrogatelab.surrogates import RBFSurrogate


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"


def main() -> None:
    """Run the β sweep and write a figure plus CSV diagnostics."""
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    problem = ReactionDiffusionProblem(pde_nx=21, mass="consistent")
    qoi = "J2"
    budgets = [10, 20, 30, 40]
    seeds = [0, 1, 2]
    samplers = [
        ("β=0", BetaGreedySampler(beta=0.0)),
        ("β=0.25", BetaGreedySampler(beta=0.25)),
        ("β=0.5", BetaGreedySampler(beta=0.5)),
        ("β=0.75", BetaGreedySampler(beta=0.75)),
        ("β=1", BetaGreedySampler(beta=1.0)),
        ("P-greedy", PGreedySampler()),
        ("f-greedy", FGreedySampler()),
    ]
    for _label, sampler in samplers:
        sampler.pool_size = 800

    forward = problem.forward(qoi)
    X_test, y_test = make_test_set(problem, qoi, per_dim=7, forward=forward)
    metric_rows = []

    for label, sampler in samplers:
        for seed in seeds:
            design, values = _build_trace(sampler, problem.dim, seed, 40, forward)
            for n_points in budgets:
                nrmse = _nrmse(design[:n_points], values[:n_points], X_test, y_test)
                metric_rows.append(
                    {
                        "kind": "nrmse",
                        "sampler": label,
                        "seed": seed,
                        "n": n_points,
                        "value": nrmse,
                    }
                )

    table = pd.DataFrame(metric_rows)
    table.to_csv(TAB / "beta_greedy_sweep.csv", index=False)
    _plot(table, FIG / "beta_greedy_sweep.pdf")

    final = table[(table["kind"] == "nrmse") & (table["n"] == 40)]
    medians = final.groupby("sampler")["value"].median()
    print("NRMSE@n=40")
    print(medians.to_string(float_format=lambda value: f"{value:.6g}"))
    print(f"CSV: {TAB / 'beta_greedy_sweep.csv'}")
    print(f"Figure: {FIG / 'beta_greedy_sweep.pdf'}")


def _build_trace(sampler, dim: int, seed: int, n_max: int, forward):
    pool = sampler._make_pool(dim)
    design = sampler._snap_to_pool(sampler._initial_design(dim, seed), pool)
    values = np.asarray(forward(design), dtype=float).ravel()
    available = ~_membership_mask(pool, design)
    sampler._block_neighbours(pool, design, available)
    state: dict = {}

    while design.shape[0] < n_max:
        surrogate = RBFSurrogate(
            kernel=sampler.scoring_kernel,
            eps=sampler.scoring_eps,
            nugget=sampler.nugget,
            log_transform=sampler.scoring_log_transform,
        ).fit(design, values)
        context = GreedyContext(
            pool=pool,
            available=available,
            X=design,
            y=values,
            surrogate=surrogate,
            iteration=design.shape[0],
            state=state,
        )
        scores = np.asarray(sampler.score(context), dtype=float).ravel()
        scores = np.where(available, scores, -np.inf)
        if not np.any(np.isfinite(scores)):
            break

        selected = int(np.argmax(scores))
        new_point = pool[selected:selected + 1]
        new_value = np.asarray(forward(new_point), dtype=float).ravel()
        sampler.after_select(context, selected, float(new_value[0]))
        design = np.vstack([design, new_point])
        values = np.append(values, new_value)
        available[selected] = False
        sampler._block_neighbours(pool, new_point, available)

    return design, values


def _nrmse(
    design: np.ndarray,
    values: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(over="ignore", invalid="ignore"):
            try:
                surrogate = RBFSurrogate(kernel="gaussian", nugget=1e-10).fit(design, values)
                prediction = surrogate.predict(X_test)
            except Exception:
                return float("inf")
    if not np.all(np.isfinite(prediction)):
        return float("inf")
    rmse = float(np.sqrt(np.mean((prediction - y_test) ** 2)))
    scale = float(np.std(y_test))
    return rmse / scale if scale > 0.0 else float("nan")


def _plot(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    metrics = table[table["kind"] == "nrmse"]
    for sampler, group in metrics.groupby("sampler"):
        curve = group.groupby("n")["value"].median().reset_index()
        ax.plot(curve["n"], curve["value"], marker="o", label=sampler)
    ax.set_yscale("log")
    ax.set_xlabel("розмір тренувального набору")
    ax.set_ylabel("NRMSE")
    ax.set_title("Розгортка β-greedy родини (P-greedy і f-greedy — крайні точки)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
