"""Generate LaTeX tables and matching CSVs comparing PIDM vs Integral method.

Reads JSON files from three verification_results* folders, picks the same 50
random sample indices (present in all three folders) using a fixed seed, and
emits one A4-width LaTeX table + one CSV per noise level.

Required LaTeX packages: booktabs, graphicx.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

# (folder_name, filename_template, noise_label, file_suffix)
FOLDERS = [
    ("verification_results",         "sample_{idx}_ddpm_vs_integral.json",            0.00, "00"),
    ("verification_results_noise005", "sample_{idx}_ddpm_vs_integral_noise0.05.json", 0.05, "005"),
    ("verification_results_noise010", "sample_{idx}_ddpm_vs_integral_noise0.1.json",  0.10, "010"),
]

IDX_RE = re.compile(r"^sample_(\d+)_ddpm_vs_integral(?:_noise[0-9.]+)?\.json$")


def collect_indices(folder: Path) -> set[int]:
    out: set[int] = set()
    if not folder.is_dir():
        return out
    for name in os.listdir(folder):
        m = IDX_RE.match(name)
        if m:
            out.add(int(m.group(1)))
    return out


def load_row(folder: Path, template: str, idx: int) -> dict:
    path = folder / template.format(idx=idx)
    with open(path) as f:
        d = json.load(f)
    m = d["metrics"]
    ddpm_rmse = m["ddpm_vs_gt"]["rmse"]
    ddpm_l2r = m["ddpm_vs_gt"]["l2_rel"]
    int_rmse = m["integral_vs_gt"]["rmse"]
    int_l2r = m["integral_vs_gt"]["l2_rel"]

    pidm_wins = (ddpm_rmse < int_rmse) + (ddpm_l2r < int_l2r)
    int_wins = (int_rmse < ddpm_rmse) + (int_l2r < ddpm_l2r)
    if pidm_wins == 2:
        winner = "PIDM"
    elif int_wins == 2:
        winner = "IEM"
    else:
        winner = "Tie"

    return {
        "sample_idx": int(d["sample_idx"]),
        "noise_level": float(d["noise_level"]),
        "ensemble_size": int(d["ensemble_size"]),
        "lam": float(d["lam"]),
        "cond_A": float(d["cond_A"]),
        "ddpm_rmse": ddpm_rmse,
        "ddpm_l2_rel": ddpm_l2r,
        "integral_rmse": int_rmse,
        "integral_l2_rel": int_l2r,
        "winner": winner,
    }


def fmt_sci(x: float, sig: int = 2) -> str:
    """Format float as LaTeX scientific notation, e.g. 5.81\\times10^{-6}."""
    if x == 0.0:
        return "0"
    s = f"{x:.{sig}e}"  # e.g. 5.81e-06
    mant, exp = s.split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def fmt4(x: float) -> str:
    return f"{x:.4f}"


def build_latex_table(rows: list[dict], noise_label: float, suffix: str) -> str:
    if noise_label == 0.0:
        caption = "PIDM vs Integral equation method"
    else:
        caption = f"PIDM vs Integral equation method (noise {noise_label:.2f})"
    label = f"tab:pidm_vs_integral_noise_{suffix}"

    header_top = (
        " & ".join([
            r"\multirow{2}{*}{№}",
            r"\multirow{2}{*}{Noise}",
            r"\multirow{2}{*}{$N_{\mathrm{ens}}$}",
            r"\multirow{2}{*}{$\alpha$}",
            r"\multirow{2}{*}{$\kappa(A)$}",
            r"\multicolumn{2}{c}{PIDM vs GT}",
            r"\multicolumn{2}{c}{Integral vs GT}",
            r"\multirow{2}{*}{Winner}",
        ])
        + r" \\"
    )
    header_cmidrules = r"\cmidrule(lr){6-7} \cmidrule(lr){8-9}"
    header_bot = (
        " & & & & & RMSE & $L^{2}_{\\mathrm{rel}}$ & RMSE & $L^{2}_{\\mathrm{rel}}$ & "
        + r" \\"
    )

    body_lines = []
    for r in rows:
        body_lines.append(
            " & ".join([
                str(r["sample_idx"]),
                f"{r['noise_level']:.2f}",
                str(r["ensemble_size"]),
                fmt_sci(r["lam"]),
                fmt_sci(r["cond_A"]),
                fmt4(r["ddpm_rmse"]),
                fmt4(r["ddpm_l2_rel"]),
                fmt4(r["integral_rmse"]),
                fmt4(r["integral_l2_rel"]),
                r["winner"],
            ])
            + r" \\"
        )

    body = "\n".join(body_lines)
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{rcrlrrrrrc}\n"
        "\\toprule\n"
        f"{header_top}\n"
        f"{header_cmidrules}\n"
        f"{header_bot}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\end{table}\n"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "sample_idx", "noise_level", "ensemble_size", "lam", "cond_A",
        "ddpm_rmse", "ddpm_l2_rel", "integral_rmse", "integral_l2_rel", "winner",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


METRIC_KEYS = ["mse", "rmse", "l2", "l2_rel", "linf", "linf_rel"]
METRIC_LABELS = {
    "mse": r"MSE",
    "rmse": r"RMSE",
    "l2": r"$L^{2}$",
    "l2_rel": r"$L^{2}_{\mathrm{rel}}$",
    "linf": r"$L^{\infty}$",
    "linf_rel": r"$L^{\infty}_{\mathrm{rel}}$",
}


def compute_winrate(folder: Path, template: str, indices: list[int]) -> dict:
    """Per-metric PIDM/IEM/Tie win counts."""
    stats = {k: {"PIDM": 0, "IEM": 0, "Tie": 0} for k in METRIC_KEYS}
    n = 0
    for idx in indices:
        path = folder / template.format(idx=idx)
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        m = d["metrics"]
        for k in METRIC_KEYS:
            p = m["ddpm_vs_gt"][k]
            i = m["integral_vs_gt"][k]
            if p < i:
                stats[k]["PIDM"] += 1
            elif i < p:
                stats[k]["IEM"] += 1
            else:
                stats[k]["Tie"] += 1
        n += 1
    return {"n": n, "metrics": stats}


def _pct(c: int, n: int) -> str:
    return f"{100.0 * c / n:.1f}\\%" if n else "--"


def build_winrate_table(stats: list) -> str:
    body_lines = []
    for i, (noise, s) in enumerate(stats):
        n = s["n"]
        for j, k in enumerate(METRIC_KEYS):
            ms = s["metrics"][k]
            noise_cell = (
                f"\\multirow{{{len(METRIC_KEYS)}}}{{*}}{{{noise:.2f}}}" if j == 0 else ""
            )
            n_cell = (
                f"\\multirow{{{len(METRIC_KEYS)}}}{{*}}{{{n}}}" if j == 0 else ""
            )
            body_lines.append(
                " & ".join([
                    noise_cell,
                    n_cell,
                    METRIC_LABELS[k],
                    f"{ms['PIDM']} ({_pct(ms['PIDM'], n)})",
                    f"{ms['IEM']} ({_pct(ms['IEM'], n)})",
                ])
                + r" \\"
            )
        if i < len(stats) - 1:
            body_lines.append(r"\midrule")
    body = "\n".join(body_lines)

    header = "Noise & $N$ & Metric & PIDM wins & IEM wins \\\\"
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Per-metric win rates of PIDM vs the Integral equation method (IEM) "
        "across all available samples, by noise level. "
        "Each cell shows the number of samples on which the corresponding method achieved "
        "the lower error, with the percentage of samples in parentheses.}\n"
        "\\label{tab:winrate_summary}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{cclrr}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\end{table}\n"
    )


def write_winrate_csv(path: Path, stats: list) -> None:
    fieldnames = [
        "noise_level", "n", "metric",
        "pidm_wins", "iem_wins", "ties",
        "pidm_pct", "iem_pct", "tie_pct",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for noise, s in stats:
            n = s["n"]
            for k in METRIC_KEYS:
                ms = s["metrics"][k]
                w.writerow({
                    "noise_level": noise,
                    "n": n,
                    "metric": k,
                    "pidm_wins": ms["PIDM"],
                    "iem_wins": ms["IEM"],
                    "ties": ms["Tie"],
                    "pidm_pct": (100.0 * ms["PIDM"] / n) if n else 0.0,
                    "iem_pct": (100.0 * ms["IEM"] / n) if n else 0.0,
                    "tie_pct": (100.0 * ms["Tie"] / n) if n else 0.0,
                })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to <root>/latex_tables")
    args = p.parse_args()

    out_dir: Path = args.out_dir or (args.root / "latex_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    folder_paths = [(args.root / f, tmpl, noise, suf) for f, tmpl, noise, suf in FOLDERS]

    # Indices common to all three folders.
    idx_sets = [collect_indices(fp) for fp, _, _, _ in folder_paths]
    common = set.intersection(*idx_sets)
    if len(common) < args.n:
        raise SystemExit(
            f"Only {len(common)} indices common to all folders, need {args.n}."
        )

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(sorted(common), args.n))
    print(f"Selected {len(selected)} indices (seed={args.seed}): "
          f"{selected[:5]} ... {selected[-5:]}")

    all_tex_parts: list[str] = []
    for folder, template, noise, suffix in folder_paths:
        rows = [load_row(folder, template, i) for i in selected]
        tex = build_latex_table(rows, noise, suffix)
        tex_path = out_dir / f"table_noise_{suffix}.tex"
        csv_path = out_dir / f"table_noise_{suffix}.csv"
        tex_path.write_text(tex)
        write_csv(csv_path, rows)
        all_tex_parts.append((suffix, tex))
        print(f"\nWrote {tex_path} and {csv_path}")
        print(tex)

    # Win-rate summary across ALL available samples per noise level.
    stats = []
    for folder, template, noise, _suf in folder_paths:
        idxs = sorted(collect_indices(folder))
        s = compute_winrate(folder, template, idxs)
        stats.append((noise, s))
        ms = s["metrics"]
        print(f"Noise {noise:.2f}: N={s['n']}")
        for k in METRIC_KEYS:
            r = ms[k]
            print(f"  {k:8s} PIDM/IEM/Tie={r['PIDM']}/{r['IEM']}/{r['Tie']}")

    winrate_tex = build_winrate_table(stats)
    (out_dir / "table_winrate.tex").write_text(winrate_tex)
    write_winrate_csv(out_dir / "table_winrate.csv", stats)
    print(f"\nWrote {out_dir / 'table_winrate.tex'} and {out_dir / 'table_winrate.csv'}")
    print(winrate_tex)

    # Combined wrapper.
    combined = (
        "% Combined comparison tables. Required LaTeX packages: booktabs, graphicx, multirow.\n"
        + "\n".join(f"\\input{{table_noise_{suf}.tex}}" for suf, _ in all_tex_parts)
        + "\n\\input{table_winrate.tex}\n"
    )
    (out_dir / "all_tables.tex").write_text(combined)
    print(f"Wrote {out_dir / 'all_tables.tex'}")


if __name__ == "__main__":
    main()
