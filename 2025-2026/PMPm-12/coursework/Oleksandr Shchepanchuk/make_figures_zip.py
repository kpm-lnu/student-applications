#!/usr/bin/env python3
"""Bundle PRIMARY figures from outputs/figures/ into figures.zip
for one-shot upload to Prism (web LaTeX editor).

Files are written at the top level of the zip (no nested folder) so
that on extraction they drop straight into the user's `figures/`
directory.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "outputs" / "figures"
DST = ROOT / "figures.zip"

PRIMARY_PDFS = [
    # §5.1 — problem descriptions
    "branin_surface.pdf",
    "heat_kappa_schema.pdf",
    # §5.4 — convergence (81-grid)
    "convergence_branin_f.pdf",
    "convergence_heat_amicon_q.pdf",
    "convergence_advdiff_amicon_q.pdf",
    "convergence_rd_2d_J.pdf",
    "convergence_rd_4d_J.pdf",
    # §5.4 — samples_to_tol (81-grid)
    "samples_to_tol_branin_f.pdf",
    "samples_to_tol_heat_amicon_q.pdf",
    "samples_to_tol_advdiff_amicon_q.pdf",
    "samples_to_tol_rd_2d_J.pdf",
    "samples_to_tol_rd_4d_J.pdf",
    # §5.4 — paper-compat
    "convergence_heat_amicon_q_paper.pdf",
    "convergence_advdiff_amicon_q_paper.pdf",
    "samples_to_tol_heat_amicon_q_paper.pdf",
    "samples_to_tol_advdiff_amicon_q_paper.pdf",
    # §5.5 — overall
    "overall_summary.pdf",
    # §4.3 — beta-greedy sweep
    "beta_greedy_sweep.pdf",
    # §5.6 — RBF vs Kriging
    "rbf_vs_kriging_rd_2d_convergence.pdf",
    "rbf_vs_kriging_rd_2d_metrics.pdf",
    # §5.2 — design panels (contrast example)
    "design_panel_heat_amicon_q_beta_greedy_0.5.pdf",
    "design_panel_heat_amicon_q_LHS.pdf",
]

PRIMARY_PNGS = [
    # §5.1 — RD illustrations
    "ode_solution.png",
    "pde_surface_y1.png",
    "spatial_phase.png",
]

ALL_FILES = PRIMARY_PDFS + PRIMARY_PNGS


def main() -> int:
    print(f"Source: {SRC}")
    print(f"Target: {DST}")
    print(f"Files expected: {len(ALL_FILES)}")
    print()

    missing: list[str] = []
    found: list[Path] = []
    for fname in ALL_FILES:
        path = SRC / fname
        if path.exists():
            found.append(path)
        else:
            missing.append(fname)

    if missing:
        print("⚠️  ВІДСУТНІ ФАЙЛИ:")
        for f in missing:
            print(f"   ✗ {f}")
        print()

    print(f"Знайдено: {len(found)}/{len(ALL_FILES)} файлів")

    if not found:
        print("Жодного файлу не знайдено — нічого пакувати.")
        return 1

    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in found:
            zf.write(path, arcname=path.name)

    size_kb = DST.stat().st_size / 1024
    print(f"\n✓ Створено {DST.name} ({size_kb:.0f} KB, {len(found)} файлів)")
    print(f"  Розташування: {DST}")

    print("\nВміст архіву:")
    with zipfile.ZipFile(DST, "r") as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            print(f"   {info.filename}  ({info.file_size // 1024} KB)")

    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
