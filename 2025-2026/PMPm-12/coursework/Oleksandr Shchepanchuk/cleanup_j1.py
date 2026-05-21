#!/usr/bin/env python3
"""Strip J1 from all reporting artefacts.

Touched files:
  outputs/summary.json
  outputs/tables/*.csv  (J1 rows removed where qoi column exists)
  outputs/figures/*_J1.pdf  (deleted)
And regenerates overall_summary.pdf + winner / speedup tables.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TAB = OUT / "tables"
FIG = OUT / "figures"


def filter_summary_json() -> None:
    p = OUT / "summary.json"
    with p.open() as f:
        s = json.load(f)

    for prob in list(s.get("qois", {}).keys()):
        s["qois"][prob] = [q for q in s["qois"][prob] if q != "J1"]

    for section in ("median_nrmse_at_max_budget", "samples_to_tolerance", "speedup_vs_LHS"):
        if section not in s:
            continue
        for prob in s[section]:
            if "J1" in s[section][prob]:
                del s[section][prob]["J1"]

    with p.open("w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    print("OK summary.json: J1 видалено")


def filter_csv_files() -> None:
    csv_files_to_filter = [
        "rd_2d_surrogate_results.csv",
        "rd_2d_surrogate_summary_nrmse.csv",
        "rd_2d_samples_to_tolerance.csv",
        "rd_4d_surrogate_results.csv",
        "rd_4d_surrogate_summary_nrmse.csv",
        "rd_4d_samples_to_tolerance.csv",
        "surrogate_results_all_problems.csv",
        "rbf_vs_kriging.csv",
        "rbf_vs_kriging_all.csv",
    ]
    total_removed = 0
    for fname in csv_files_to_filter:
        p = TAB / fname
        if not p.exists():
            print(f"  skip {fname} (not found)")
            continue
        df = pd.read_csv(p)
        if "qoi" not in df.columns:
            print(f"  skip {fname} (no qoi column)")
            continue
        before = len(df)
        df = df[df["qoi"] != "J1"]
        after = len(df)
        df.to_csv(p, index=False)
        print(f"  OK {fname}: {before} → {after} ({before - after} J1 rows removed)")
        total_removed += before - after
    print(f"  Total J1 rows removed: {total_removed}")


def remove_j1_pdfs() -> None:
    pdfs = [
        "convergence_rd_2d_J1.pdf",
        "convergence_rd_4d_J1.pdf",
        "samples_to_tol_rd_2d_J1.pdf",
        "samples_to_tol_rd_4d_J1.pdf",
    ]
    n_removed = 0
    for fname in pdfs:
        p = FIG / fname
        if p.exists():
            p.unlink()
            print(f"  OK deleted {fname}")
            n_removed += 1
        else:
            print(f"  - {fname} not found (already absent)")
    print(f"  Total PDFs removed: {n_removed}")


def regenerate_overall_summary() -> None:
    from surrogatelab import plotting as plot
    with (OUT / "summary.json").open() as f:
        s = json.load(f)
    plot.plot_overall_summary(s, str(FIG / "overall_summary.pdf"))
    print("OK overall_summary.pdf перегенеровано без J1")


def regenerate_winner_table() -> None:
    result = subprocess.run(
        [".venv/bin/python", "diagnostics_winner_table.py"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if result.returncode == 0:
        print("OK winner_per_problem.csv і speedup_vs_lhs.csv оновлено")
    else:
        print(f"  FAIL: {result.stderr}")


def main() -> int:
    print("=== Cleanup J1 ===\n")
    print("1. summary.json:")
    filter_summary_json()
    print("\n2. CSV files:")
    filter_csv_files()
    print("\n3. J1 PDF files:")
    remove_j1_pdfs()
    print("\n4. overall_summary.pdf:")
    regenerate_overall_summary()
    print("\n5. winner_table:")
    regenerate_winner_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
