"""Regenerate convergence + overall-summary figures from saved CSVs
with Ukrainian labels (plotting.py was edited after the run started, so
the running process used cached English-labelled functions)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from surrogatelab import plotting as plot
from surrogatelab.experiment import samples_to_tolerance, summarise

ROOT = Path(__file__).resolve().parent
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"

SLUGS = {
    "Branin": "branin",
    "Heat-AMICon": "heat_amicon",
    "AdvDiff-AMICon": "advdiff_amicon",
    "RD-2D": "rd_2d",
    "RD-4D": "rd_4d",
}
TOLERANCES = {
    "Branin": 0.05,
    "Heat-AMICon": 0.10,
    "AdvDiff-AMICon": 0.10,
    "RD-2D": 0.20,
    "RD-4D": 0.20,
}
N_MAX = {
    "Branin": 60,
    "Heat-AMICon": 80,
    "AdvDiff-AMICon": 80,
    "RD-2D": 60,
    "RD-4D": 150,
}


def main() -> int:
    for label, slug in SLUGS.items():
        results_path = TAB / f"{slug}_surrogate_results.csv"
        if not results_path.exists():
            print(f"skip {label}: {results_path} missing")
            continue
        results = pd.read_csv(results_path)
        summary = summarise(results, "nrmse")

        for qoi in results["qoi"].unique():
            plot.plot_convergence(
                summary,
                qoi,
                "nrmse",
                str(FIG / f"convergence_{slug}_{qoi}.pdf"),
                ylabel="NRMSE",
            )
            tol_rows = samples_to_tolerance(
                results[results["qoi"] == qoi], TOLERANCES[label]
            )
            plot.plot_samples_to_tol(
                tol_rows,
                qoi,
                TOLERANCES[label],
                N_MAX[label],
                str(FIG / f"samples_to_tol_{slug}_{qoi}.pdf"),
            )
        print(f"replotted {label}")

    summary_json_path = ROOT / "outputs" / "summary.json"
    if summary_json_path.exists():
        with summary_json_path.open() as fh:
            payload = json.load(fh)
        plot.plot_overall_summary(payload, str(FIG / "overall_summary.pdf"))
        print("replotted overall_summary")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
