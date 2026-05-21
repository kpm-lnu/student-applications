#!/usr/bin/env python3
"""Build outputs/MANIFEST.md - artefact map for §5 of the thesis.

Classifies every PDF/PNG/CSV by purpose:
  PRIMARY    — used in the thesis text (~14-15 files)
  SECONDARY  — reserve, for committee questions
  DIAGNOSTIC — for research analysis
  UNCLASSIFIED — needs manual review
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"


PRIMARY_PDFS = {
    "branin_surface.pdf": "§5.1 ілюстрація задачі Бренін",
    "heat_kappa_schema.pdf": "§5.1 схема κ(x) для Heat-AMICon",
    "convergence_branin_f.pdf": "§5.4 збіжність — Branin/f",
    "convergence_heat_amicon_q.pdf": "§5.4 збіжність — Heat-AMICon/q",
    "convergence_advdiff_amicon_q.pdf": "§5.4 збіжність — AdvDiff/q",
    "convergence_rd_2d_J.pdf": "§5.4 збіжність — RD-2D/J",
    "convergence_rd_4d_J.pdf": "§5.4 збіжність — RD-4D/J",
    "samples_to_tol_branin_f.pdf": "§5.4 бюджет — Branin/f",
    "samples_to_tol_heat_amicon_q.pdf": "§5.4 бюджет — Heat-AMICon/q",
    "samples_to_tol_advdiff_amicon_q.pdf": "§5.4 бюджет — AdvDiff/q",
    "samples_to_tol_rd_2d_J.pdf": "§5.4 бюджет — RD-2D/J",
    "samples_to_tol_rd_4d_J.pdf": "§5.4 бюджет — RD-4D/J",
    "overall_summary.pdf": "§5.5 зведене порівняння",
    "beta_greedy_sweep.pdf": "§4.3 sweep по β",
    "rbf_vs_kriging_rd_2d_convergence.pdf": "§5.6 RBF vs Kriging — основне",
    "rbf_vs_kriging_rd_2d_metrics.pdf": "§5.6 RBF vs Kriging — метрики",
    "convergence_heat_amicon_q_paper.pdf": "§5.4 paper-compat — Heat",
    "convergence_advdiff_amicon_q_paper.pdf": "§5.4 paper-compat — AdvDiff",
    "samples_to_tol_heat_amicon_q_paper.pdf": "§5.4 paper-compat — Heat бюджет",
    "samples_to_tol_advdiff_amicon_q_paper.pdf": "§5.4 paper-compat — AdvDiff бюджет",
}

PRIMARY_PNGS = {
    "ode_solution.png": "§5.1 ODE-розв'язок для RD",
    "pde_surface_y1.png": "§5.1 PDE-поверхня y1 для RD",
    "spatial_phase.png": "§5.1 просторова фаза для RD",
}

PRIMARY_DESIGN_PANELS = {
    "design_panel_heat_amicon_q_beta_greedy_0.5.pdf": "§5.2 β-greedy на Heat (адаптивний)",
    "design_panel_heat_amicon_q_LHS.pdf": "§5.2 LHS на Heat (неадаптивний, для контрасту)",
}

PRIMARY_CSVS = {
    "winner_per_problem.csv": "§5.5 переможці по кожній задачі",
    "speedup_vs_lhs.csv": "§5.5 прискорення vs LHS",
    "beta_greedy_sweep.csv": "§4.3 sweep по β дані",
    "branin_surrogate_summary_nrmse.csv": "§5.4 NRMSE — Branin",
    "heat_amicon_surrogate_summary_nrmse.csv": "§5.4 NRMSE — Heat-AMICon",
    "advdiff_amicon_surrogate_summary_nrmse.csv": "§5.4 NRMSE — AdvDiff",
    "rd_2d_surrogate_summary_nrmse.csv": "§5.4 NRMSE — RD-2D",
    "rd_4d_surrogate_summary_nrmse.csv": "§5.4 NRMSE — RD-4D",
    "heat_amicon_paper_samples_to_tol.csv": "§5.4 paper-compat hits — Heat",
    "advdiff_amicon_paper_samples_to_tol.csv": "§5.4 paper-compat hits — AdvDiff",
}

SECONDARY_PDFS = {
    "convergence_rd_2d_J2.pdf": "RD-2D/J2 — допоміжна QoI",
    "convergence_rd_4d_J2.pdf": "RD-4D/J2 — допоміжна QoI",
    "samples_to_tol_rd_2d_J2.pdf": "RD-2D/J2 — бюджет",
    "samples_to_tol_rd_4d_J2.pdf": "RD-4D/J2 — бюджет",
    "rbf_vs_kriging_branin_convergence.pdf": "RBF vs Kriging — Branin",
    "rbf_vs_kriging_branin_metrics.pdf": "RBF vs Kriging — Branin метрики",
    "rbf_vs_kriging_heat_amicon_convergence.pdf": "RBF vs Kriging — Heat",
    "rbf_vs_kriging_heat_amicon_metrics.pdf": "RBF vs Kriging — Heat метрики",
    "rbf_vs_kriging_convergence.pdf": "RBF vs Kriging (legacy)",
    "rbf_vs_kriging_metrics.pdf": "RBF vs Kriging (legacy)",
}

DIAGNOSTIC_OTHER = {
    "pde_traces.png": "RD — сліди на межах",
    "pde_surface_y2.png": "RD — поверхня y2",
    "ode_phase_portrait.png": "RD — фазовий портрет ODE",
}

DIAGNOSTIC_CSVS = {
    "equilibria.csv": "RD — рівноваги",
    "functionals_base.csv": "RD — функціонали для базових параметрів",
    "rbf_vs_kriging.csv": "RBF vs Kriging legacy CSV",
    "rbf_vs_kriging_all.csv": "RBF vs Kriging об'єднана таблиця",
    "branin_surrogate_results.csv": "raw results — Branin",
    "heat_amicon_surrogate_results.csv": "raw results — Heat",
    "advdiff_amicon_surrogate_results.csv": "raw results — AdvDiff",
    "rd_2d_surrogate_results.csv": "raw results — RD-2D",
    "rd_4d_surrogate_results.csv": "raw results — RD-4D",
    "branin_samples_to_tolerance.csv": "raw tol table — Branin",
    "heat_amicon_samples_to_tolerance.csv": "raw tol table — Heat",
    "advdiff_amicon_samples_to_tolerance.csv": "raw tol table — AdvDiff",
    "rd_2d_samples_to_tolerance.csv": "raw tol table — RD-2D",
    "rd_4d_samples_to_tolerance.csv": "raw tol table — RD-4D",
    "surrogate_results_all_problems.csv": "combined raw results",
    "heat_amicon_paper_results.csv": "raw paper-compat — Heat",
    "advdiff_amicon_paper_results.csv": "raw paper-compat — AdvDiff",
    "heat_amicon_paper_summary_nrmse.csv": "paper-compat summary — Heat",
    "advdiff_amicon_paper_summary_nrmse.csv": "paper-compat summary — AdvDiff",
}


def section_of(desc: str) -> str:
    return desc.split(" ")[0] if desc.startswith("§") else "—"


def main() -> int:
    pdf_files = {p.name for p in FIG.glob("*.pdf")}
    png_files = {p.name for p in FIG.glob("*.png")}
    csv_files = {p.name for p in TAB.glob("*.csv")}

    lines: list[str] = []
    lines.append("# Карта артефактів для §5 курсової\n\n")
    lines.append("Згенеровано автоматично з `build_manifest.py`.\n\n")
    lines.append(
        f"**Всього файлів:** {len(pdf_files)} PDF + {len(png_files)} PNG "
        f"+ {len(csv_files)} CSV\n\n"
    )
    lines.append("---\n\n")

    listed_pdf: set[str] = set()
    listed_png: set[str] = set()
    listed_csv: set[str] = set()

    lines.append("## PRIMARY — у тексті курсової\n\n")
    lines.append("Ці файли вставляються в LaTeX через `\\includegraphics{...}`.\n\n")
    lines.append("### Графіки PDF\n\n")
    lines.append("| Файл | Підрозділ | Призначення |\n|---|---|---|\n")
    for fname, desc in PRIMARY_PDFS.items():
        if fname in pdf_files:
            lines.append(f"| `{fname}` | {section_of(desc)} | {desc} |\n")
            listed_pdf.add(fname)
        else:
            lines.append(f"| `{fname}` | — | ⚠️ ВІДСУТНІЙ: {desc} |\n")

    lines.append("\n### Графіки PNG (для §5.1)\n\n")
    lines.append("| Файл | Підрозділ | Призначення |\n|---|---|---|\n")
    for fname, desc in PRIMARY_PNGS.items():
        if fname in png_files:
            lines.append(f"| `{fname}` | {section_of(desc)} | {desc} |\n")
            listed_png.add(fname)
        else:
            lines.append(f"| `{fname}` | — | ⚠️ ВІДСУТНІЙ: {desc} |\n")

    lines.append("\n### Design panels (приклади для §5.2)\n\n")
    lines.append("| Файл | Підрозділ | Призначення |\n|---|---|---|\n")
    for fname, desc in PRIMARY_DESIGN_PANELS.items():
        if fname in pdf_files:
            lines.append(f"| `{fname}` | {section_of(desc)} | {desc} |\n")
            listed_pdf.add(fname)
        else:
            lines.append(f"| `{fname}` | — | ⚠️ ВІДСУТНІЙ: {desc} |\n")

    lines.append("\n### Таблиці CSV\n\n")
    lines.append("| Файл | Підрозділ | Призначення |\n|---|---|---|\n")
    for fname, desc in PRIMARY_CSVS.items():
        if fname in csv_files:
            lines.append(f"| `{fname}` | {section_of(desc)} | {desc} |\n")
            listed_csv.add(fname)
        else:
            lines.append(f"| `{fname}` | — | ⚠️ ВІДСУТНІЙ: {desc} |\n")

    lines.append("\n## SECONDARY — резерв (НЕ у тексті, для питань комісії)\n\n")
    lines.append("### Графіки PDF\n\n")
    lines.append("| Файл | Призначення |\n|---|---|\n")
    for fname, desc in SECONDARY_PDFS.items():
        if fname in pdf_files:
            lines.append(f"| `{fname}` | {desc} |\n")
            listed_pdf.add(fname)

    other_panels = sorted(
        f for f in pdf_files
        if f.startswith("design_panel_") and f not in listed_pdf
    )
    lines.append(
        f"\n**+{len(other_panels)} додаткових design panels** "
        f"(всі комбінації задача×семплер):\n\n"
    )
    for f in other_panels:
        lines.append(f"- `{f}`\n")
        listed_pdf.add(f)

    lines.append("\n## DIAGNOSTIC — для дослідницького аналізу\n\n")
    lines.append("Сирі дані для повторюваності, не для тексту.\n\n")
    for fname, desc in DIAGNOSTIC_OTHER.items():
        if fname in png_files:
            lines.append(f"- `{fname}` — {desc}\n")
            listed_png.add(fname)
    for fname, desc in DIAGNOSTIC_CSVS.items():
        if fname in csv_files:
            lines.append(f"- `{fname}` — {desc}\n")
            listed_csv.add(fname)

    unclassified_pdf = pdf_files - listed_pdf
    unclassified_png = png_files - listed_png
    unclassified_csv = csv_files - listed_csv
    if unclassified_pdf or unclassified_png or unclassified_csv:
        lines.append("\n## ⚠️ НЕКЛАСИФІКОВАНІ — перевірити вручну\n\n")
        for f in sorted(unclassified_pdf):
            lines.append(f"- PDF: `{f}`\n")
        for f in sorted(unclassified_png):
            lines.append(f"- PNG: `{f}`\n")
        for f in sorted(unclassified_csv):
            lines.append(f"- CSV: `{f}`\n")

    summary_path = OUT / "summary.json"
    if summary_path.exists():
        with summary_path.open() as f:
            s = json.load(f)
        lines.append("\n---\n\n## Підсумок NRMSE після rerun (без J1)\n\n")
        lines.append(
            "| Задача / QoI | Найкращий | NRMSE | LHS NRMSE | "
            "Покращення | Толер. |\n"
        )
        lines.append("|---|---|---|---|---|---|\n")
        for prob in s["problems"]:
            tol = s.get("tolerances", {}).get(prob, "—")
            for qoi in s["qois"][prob]:
                nrmse = s["median_nrmse_at_max_budget"][prob][qoi]
                best = min(nrmse.items(), key=lambda kv: kv[1])
                lhs = nrmse.get("LHS", float("nan"))
                improvement = (lhs / best[1]) if best[1] > 0 else float("nan")
                hit_tol = "✓" if best[1] < tol else "✗" if isinstance(tol, (int, float)) else "—"
                lines.append(
                    f"| {prob} / {qoi} | {best[0]} | {best[1]:.3f} {hit_tol} | "
                    f"{lhs:.3f} | {improvement:.1f}× | τ={tol} |\n"
                )

        lines.append("\n## Samples-to-tolerance hits\n\n")
        lines.append("Скільки семплерів досягли τ для кожної задачі:\n\n")
        lines.append("| Задача / QoI | Hit | Miss |\n|---|---|---|\n")
        for prob in s["problems"]:
            for qoi in s["qois"][prob]:
                sto = s["samples_to_tolerance"][prob][qoi]
                hits = sum(1 for n in sto.values() if n is not None)
                misses = sum(1 for n in sto.values() if n is None)
                lines.append(f"| {prob} / {qoi} | {hits}/8 | {misses}/8 |\n")

    out_path = OUT / "MANIFEST.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"OK {out_path}")
    print(f"  PDF: {len(pdf_files)} total ({len(unclassified_pdf)} unclassified)")
    print(f"  PNG: {len(png_files)} total ({len(unclassified_png)} unclassified)")
    print(f"  CSV: {len(csv_files)} total ({len(unclassified_csv)} unclassified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
