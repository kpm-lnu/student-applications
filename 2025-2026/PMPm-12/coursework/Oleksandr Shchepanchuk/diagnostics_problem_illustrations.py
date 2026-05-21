#!/usr/bin/env python3
"""Illustrative figures for §5.1 (test problem descriptions).

Generates:
  * branin_surface.pdf  - log10(Branin+1) contour with 3 known minima
  * heat_kappa_schema.pdf - piecewise-constant kappa diagram
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-surrogatelab")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from surrogatelab.problems import BraninProblem

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def plot_branin_surface() -> Path:
    problem = BraninProblem()
    x1 = np.linspace(-5.0, 10.0, 200)
    x2 = np.linspace(0.0, 15.0, 200)
    X1, X2 = np.meshgrid(x1, x2)
    pts_phys = np.column_stack([X1.ravel(), X2.ravel()])
    pts_unit = problem.to_unit(pts_phys)
    Z = problem.forward("f")(pts_unit).reshape(X1.shape)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    cf = ax.contourf(X1, X2, np.log10(Z + 1.0), levels=20, cmap="viridis")
    minima = [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]
    for mx, my in minima:
        ax.scatter([mx], [my], c="red", marker="*", s=220,
                   edgecolor="white", linewidth=1.5, zorder=10)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(r"Функція Бренін: $\log_{10}(f+1)$ і три глобальні мінімуми")
    plt.colorbar(cf, ax=ax, label=r"$\log_{10}(f+1)$")
    plt.tight_layout()
    out = FIG / "branin_surface.pdf"
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {out.name}")
    return out


def plot_heat_kappa_schema() -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x_left = np.linspace(0.0, 0.5, 50)
    x_right = np.linspace(0.5, 1.0, 50)
    ax.plot(x_left, np.full_like(x_left, 2.5), color="#1f77b4", lw=3,
            label=r"$\kappa = \mu_1$")
    ax.plot(x_right, np.full_like(x_right, 7.0), color="#d62728", lw=3,
            label=r"$\kappa = \mu_2$")
    ax.axvline(0.5, color="k", linestyle="--", alpha=0.5)
    ax.text(0.25, 2.5 + 0.4, r"$\mu_1$", ha="center", fontsize=14,
            color="#1f77b4")
    ax.text(0.75, 7.0 + 0.4, r"$\mu_2$", ha="center", fontsize=14,
            color="#d62728")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\kappa(x)$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_title(r"Кусково-стала теплопровідність на $[0,\,1]$")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG / "heat_kappa_schema.pdf"
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  OK {out.name}")
    return out


def main() -> int:
    plot_branin_surface()
    plot_heat_kappa_schema()
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
