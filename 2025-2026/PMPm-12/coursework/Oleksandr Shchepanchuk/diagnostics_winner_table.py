#!/usr/bin/env python3
"""Build winner_per_problem.csv and speedup_vs_lhs.csv from summary.json.

These are convenience tables for §5.5 (per-problem winner at n_max) and
§6 (speed-up over LHS in samples-to-tolerance).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TAB = ROOT / "outputs" / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def main() -> int:
    with open(ROOT / "outputs" / "summary.json") as fh:
        s = json.load(fh)

    # Winner per (problem, qoi) at n_max
    winner_rows = []
    for problem in s["problems"]:
        for qoi in s["qois"][problem]:
            nrmse_dict = s["median_nrmse_at_max_budget"][problem][qoi]
            winner = min(nrmse_dict.items(), key=lambda kv: kv[1])
            lhs_nrmse = nrmse_dict.get("LHS", float("nan"))
            improvement = (
                round(lhs_nrmse / winner[1], 2)
                if winner[1] > 0 and lhs_nrmse > 0
                else None
            )
            winner_rows.append({
                "problem": problem,
                "qoi": qoi,
                "winner_sampler": winner[0],
                "winner_nrmse": round(winner[1], 4),
                "lhs_nrmse": round(lhs_nrmse, 4),
                "improvement_vs_lhs": improvement,
            })

    df_winner = pd.DataFrame(winner_rows)
    out_w = TAB / "winner_per_problem.csv"
    df_winner.to_csv(out_w, index=False)
    print(f"OK {out_w.name} ({len(df_winner)} rows)")
    print(df_winner.to_string(index=False))

    # Speedup vs LHS (samples-to-tolerance)
    speedup_rows = []
    for problem in s["problems"]:
        for qoi in s["qois"][problem]:
            sto = s["samples_to_tolerance"][problem][qoi]
            lhs_n = sto.get("LHS")
            for sampler, n in sto.items():
                if sampler == "LHS":
                    continue
                if lhs_n is None or n is None or n == 0:
                    continue
                speedup_rows.append({
                    "problem": problem,
                    "qoi": qoi,
                    "sampler": sampler,
                    "n_to_tol": n,
                    "lhs_n_to_tol": lhs_n,
                    "speedup": round(lhs_n / n, 2),
                })

    df_speed = pd.DataFrame(speedup_rows)
    out_s = TAB / "speedup_vs_lhs.csv"
    df_speed.to_csv(out_s, index=False)
    print(f"\nOK {out_s.name} ({len(df_speed)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
