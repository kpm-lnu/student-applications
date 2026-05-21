#!/usr/bin/env python3
"""Side experiment: RBF vs Kriging on three problems (RD-2D / Branin /
Heat-AMICon).

Both surrogates are fit on identical LHS designs of growing size with
several seeds, errors / fit-time compared on a shared 500-point test
set. Per-problem PDFs (rbf_vs_kriging_<slug>_convergence.pdf and
rbf_vs_kriging_<slug>_metrics.pdf), plus a combined CSV
`rbf_vs_kriging_all.csv`. The single-problem RD-2D outputs
(rbf_vs_kriging.csv, rbf_vs_kriging_convergence.pdf,
rbf_vs_kriging_metrics.pdf) are preserved for backward compatibility
with §5.6 of the current draft.
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
import pandas as pd

from surrogatelab.metrics import compute_metrics
from surrogatelab.problems import (
    BraninProblem,
    HeatAMICon_FD,
    ReactionDiffusionProblem,
)
from surrogatelab.sampling import get_sampler
from surrogatelab.surrogates import KrigingSurrogate, RBFSurrogate


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

N_GRID = [10, 20, 30, 40, 50]
SEEDS = list(range(10))
TEST_SIZE = 500
TEST_SEED = 0
KRIGING_N_RESTARTS = 15  # match production default

# (slug, label, factory, qoi, log_transform)
PROBLEMS = [
    ("rd_2d", "RD-2D / J", lambda: ReactionDiffusionProblem(pde_nx=41), "J", True),
    ("branin", "Branin / f", lambda: BraninProblem(), "f", False),
    ("heat_amicon", "Heat-AMICon / q", lambda: HeatAMICon_FD(n_grid=51), "q", True),
]


def main() -> None:
    """Run LHS-only RBF vs Kriging on three problems and save artefacts."""
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for slug, label, factory, qoi, log_transform in PROBLEMS:
        print(f"\n[{slug}] {label}  (log_transform={log_transform})")
        t0 = time.perf_counter()
        problem = factory()
        forward = problem.forward(qoi)

        rng = np.random.default_rng(TEST_SEED)
        X_test = rng.uniform(0.0, 1.0, (TEST_SIZE, problem.dim))
        y_test = forward(X_test)

        rows = []
        for n in N_GRID:
            for seed in SEEDS:
                X = get_sampler("LHS").build(n, problem.dim, seed=seed)
                y = forward(X)
                rows.append(_fit_and_score(
                    "RBF", X, y, X_test, y_test, n, seed,
                    slug=slug, qoi=qoi, log_transform=log_transform,
                ))
                rows.append(_fit_and_score(
                    "Kriging", X, y, X_test, y_test, n, seed,
                    slug=slug, qoi=qoi, log_transform=log_transform,
                ))

        table = pd.DataFrame(rows)
        all_rows.extend(rows)

        # Per-problem outputs
        _plot_convergence(
            table, FIG / f"rbf_vs_kriging_{slug}_convergence.pdf", label,
        )
        _plot_metrics_bars(
            table, FIG / f"rbf_vs_kriging_{slug}_metrics.pdf",
            n=max(N_GRID), label=label,
        )

        # Backward-compat single-problem RD-2D files
        if slug == "rd_2d":
            table.to_csv(TAB / "rbf_vs_kriging.csv", index=False)
            _plot_convergence(
                table, FIG / "rbf_vs_kriging_convergence.pdf", label,
            )
            _plot_metrics_bars(
                table, FIG / "rbf_vs_kriging_metrics.pdf",
                n=max(N_GRID), label=label,
            )

        summary = (
            table.groupby(["surrogate", "n"])
            .agg(median_nrmse=("nrmse", "median"),
                 median_time=("fit_time_s", "median"))
            .reset_index()
        )
        print(summary.to_string(index=False,
                                 float_format=lambda v: f"{v:.4g}"))
        print(f"  elapsed: {time.perf_counter() - t0:.1f}s")

    combined = pd.DataFrame(all_rows)
    combined.to_csv(TAB / "rbf_vs_kriging_all.csv", index=False)
    print(f"\nCombined CSV: {TAB / 'rbf_vs_kriging_all.csv'} ({len(combined)} rows)")


def _fit_and_score(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n: int,
    seed: int,
    slug: str,
    qoi: str,
    log_transform: bool,
) -> dict:
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if name == "RBF":
                model = RBFSurrogate(
                    kernel="gaussian",
                    eps=None,
                    nugget=1e-10,
                    log_transform=log_transform,
                ).fit(X, y)
            else:
                model = KrigingSurrogate(
                    nugget=1e-10,
                    log_transform=log_transform,
                    n_restarts=KRIGING_N_RESTARTS,
                ).fit(X, y)
            y_pred = model.predict(X_test)
            elapsed = time.perf_counter() - started
            metrics = compute_metrics(y_test, y_pred)
            record = metrics.as_dict()
            record.update(
                surrogate=name,
                sampler="LHS",
                problem=slug,
                qoi=qoi,
                n=n,
                seed=seed,
                fit_time_s=elapsed,
                failed=False,
            )
            return record
        except Exception as exc:
            elapsed = time.perf_counter() - started
            return {
                "surrogate": name,
                "sampler": "LHS",
                "problem": slug,
                "qoi": qoi,
                "n": n,
                "seed": seed,
                "nrmse": float("nan"),
                "mae": float("nan"),
                "r2": float("nan"),
                "max_re": float("nan"),
                "fit_time_s": elapsed,
                "failed": True,
                "error": str(exc),
            }


def _plot_convergence(table: pd.DataFrame, path: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    palette = {"RBF": "#d62728", "Kriging": "#1f77b4"}
    for name, group in table.groupby("surrogate"):
        curve = (
            group.groupby("n")["nrmse"]
            .agg(median="median",
                 q25=lambda values: np.nanpercentile(values, 25),
                 q75=lambda values: np.nanpercentile(values, 75))
            .reset_index()
        )
        ax.plot(curve["n"], curve["median"], marker="o", color=palette[name],
                label=name)
    ax.set_yscale("log")
    ax.set_xlabel("розмір тренувального набору $n_s$")
    ax.set_ylabel("NRMSE")
    ax.set_title(f"RBF vs Kriging на {label} (LHS-дизайн)")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def _plot_metrics_bars(table: pd.DataFrame, path: Path, n: int, label: str) -> None:
    sub = table[table["n"] == n]
    metrics = [("nrmse", "NRMSE"), ("mae", "MAE"),
               ("max_re", "MAX_RE"), ("r2", "$R^2$")]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), constrained_layout=True)
    names = ["RBF", "Kriging"]
    palette = {"RBF": "#d62728", "Kriging": "#1f77b4"}
    for ax, (key, ttl) in zip(axes, metrics):
        medians = [
            float(np.nanmedian(sub[sub["surrogate"] == n_][key]))
            for n_ in names
        ]
        ax.bar(np.arange(len(names)), medians,
               color=[palette[n_] for n_ in names])
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(ttl)
        if key != "r2":
            ax.set_yscale("log")
    fig.suptitle(f"Якість RBF vs Kriging при $n_s={n}$ на {label}")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
