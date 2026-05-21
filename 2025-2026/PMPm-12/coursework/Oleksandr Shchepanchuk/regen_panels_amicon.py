#!/usr/bin/env python3
"""Regenerate design_panel PDFs for Heat-AMICon and AdvDiff-AMICon
after the n_max=120 rerun."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-surrogatelab")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from diagnostics_design_panels import (  # noqa: E402
    SAMPLERS,
    build_qoi_background,
    plot_panel,
)
from surrogatelab.problems import AdvDiffAMICon_FD, HeatAMICon_FD  # noqa: E402

PROBLEMS_2D = [
    ("heat_amicon", "q", lambda: HeatAMICon_FD(n_grid=51), True),
    ("advdiff_amicon", "q", lambda: AdvDiffAMICon_FD(n_grid=51), True),
]


def main() -> int:
    n_done = 0
    for slug, qoi, factory, log_scale in PROBLEMS_2D:
        problem = factory()
        t0 = time.perf_counter()
        G1, G2, field, ls = build_qoi_background(problem, qoi, log_scale)
        print(f"[{slug}] background ready in {time.perf_counter() - t0:.1f}s")
        for sampler_name in SAMPLERS:
            try:
                out = plot_panel(slug, qoi, problem, sampler_name, G1, G2, field, ls)
                print(f"  OK {out.name}")
                n_done += 1
            except Exception as exc:
                print(f"  FAIL {slug}/{sampler_name}: {exc}")
    print(f"\nDone: {n_done}/{len(PROBLEMS_2D) * len(SAMPLERS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
