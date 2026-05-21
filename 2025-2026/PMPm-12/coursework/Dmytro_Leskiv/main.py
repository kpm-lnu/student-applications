from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import METHOD_LABELS, SCENARIO_LABELS, ModelConfig
from simulation import (
    build_scenario_comparison,
    run_monte_carlo,
    run_parameter_sweep,
    summarize_results,
)

# ── Стиль ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":   150,
    "savefig.dpi":  300,
    "axes.grid":    True,
    "grid.alpha":   0.3,
})

METHOD_COLORS = {
    "M1_Класичний_одноконтурний":   "#4C72B0",
    "M2_Класичний_багатоконтурний": "#DD8452",
    "M3_Гібридний_постквантовий":   "#55A868",
}


def _label(method_name: str) -> str:
    return METHOD_LABELS.get(method_name, method_name)


def ensure_output_dirs(base_dir: str):
    root     = Path(base_dir)
    tables   = root / "tables"
    figures  = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  збережено: {path.name}")


# ── Графіки ──────────────────────────────────────────────────────────────────

def plot_integral_scores(summary: pd.DataFrame, figures_dir: Path) -> None:
    """Стовпчиковий графік середнього J_i ± std для базового сценарію."""
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = summary["method"].tolist()
    x_pos   = np.arange(len(methods))
    colors  = [METHOD_COLORS.get(m, "#999") for m in methods]

    bars = ax.bar(x_pos, summary["integral_score_mean"], color=colors,
                  yerr=summary["integral_score_std"], capsize=5, width=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([_label(m) for m in methods], rotation=10, ha="right")
    ax.set_ylabel("Інтегральний показник $J_i$")
    ax.set_title("Середній інтегральний показник методів (базовий сценарій)")
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, summary["integral_score_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(figures_dir / "integral_scores_bar.png")
    plt.close(fig)


def plot_sweep_line(
    sweep_df: pd.DataFrame,
    metric_col: str,
    figures_dir: Path,
    filename: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Лінійний графік sweep-аналізу з довірчими інтервалами."""
    fig, ax = plt.subplots(figsize=(8, 5))
    std_col = metric_col.replace("_mean", "_std") if "_mean" in metric_col else None

    for method in sweep_df["method"].unique():
        sub = sweep_df[sweep_df["method"] == method].sort_values("sweep_value")
        lbl = _label(method)
        clr = METHOD_COLORS.get(method, None)
        ax.plot(sub["sweep_value"], sub[metric_col], marker="o", label=lbl, color=clr)
        if std_col and std_col in sub.columns:
            ax.fill_between(
                sub["sweep_value"],
                sub[metric_col] - sub[std_col],
                sub[metric_col] + sub[std_col],
                alpha=0.15, color=clr,
            )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / filename)
    plt.close(fig)


def plot_delay_energy(summary: pd.DataFrame, figures_dir: Path) -> None:
    """Порівняльний стовпчиковий графік затримки та енерговитрат."""
    methods = summary["method"].tolist()
    x_pos   = np.arange(len(methods))   # ВИПРАВЛЕНО: було 'x', конфліктувало з numpy
    width   = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos - width / 2, summary["delay_mean"],       width=width, label="Затримка")
    ax.bar(x_pos + width / 2, summary["energy_cost_mean"], width=width, label="Енерговитрати")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([_label(m) for m in methods], rotation=10, ha="right")
    ax.set_ylabel("Середнє значення")
    ax.set_title("Затримка та енерговитрати методів (базовий сценарій)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "delay_energy_comparison.png")
    plt.close(fig)


def plot_radar_chart(summary: pd.DataFrame, figures_dir: Path) -> None:
    """Радарний графік нормованих критеріїв."""
    metrics = [
        "security_score_norm_mean",
        "post_quantum_score_norm_mean",
        "risk_norm_mean",
        "delay_norm_mean",
        "computational_cost_norm_mean",
        "energy_cost_norm_mean",
        "human_factor_score_norm_mean",
        "overhead_norm_mean",
    ]
    labels = [
        "Безпека", "Постквантова\nстійкість", "Ризик",
        "Затримка", "Обч.\nвитрати", "Енерго-\nвитрати",
        "Людський\nчинник", "Накладні\nвитрати",
    ]

    # Перевіряємо наявність колонок
    missing = [m for m in metrics if m not in summary.columns]
    if missing:
        print(f"  [radar] пропущені колонки: {missing} — пропускаємо")
        return

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for _, row in summary.iterrows():
        vals = [float(row[m]) for m in metrics] + [float(row[metrics[0]])]
        clr  = METHOD_COLORS.get(row["method"], None)
        ax.plot(angles, vals, linewidth=2, label=_label(row["method"]), color=clr)
        ax.fill(angles, vals, alpha=0.07, color=clr)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Багатокритеріальне порівняння методів", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "radar_chart.png")
    plt.close(fig)


def plot_scenario_comparison(comparison_df: pd.DataFrame, figures_dir: Path) -> None:
    """Групований графік J_i для всіх сценаріїв."""
    scenarios = comparison_df["scenario"].unique()
    methods   = comparison_df["method"].unique()
    n_sc      = len(scenarios)
    n_m       = len(methods)
    width     = 0.22

    fig, ax = plt.subplots(figsize=(11, 5))
    x_pos   = np.arange(n_sc)

    for i, method in enumerate(methods):
        sub    = comparison_df[comparison_df["method"] == method]
        vals   = [sub[sub["scenario"] == sc]["integral_score_mean"].values[0]
                  if len(sub[sub["scenario"] == sc]) else 0.0 for sc in scenarios]
        offset = (i - n_m / 2 + 0.5) * width
        ax.bar(x_pos + offset, vals, width=width,
               label=_label(method), color=METHOD_COLORS.get(method))

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios],
                       rotation=12, ha="right")
    ax.set_ylabel("Середній інтегральний показник $\\bar{J}_i$")
    ax.set_title("Порівняння методів у різних сценаріях")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "scenario_comparison.png")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    config = ModelConfig()
    tables_dir, figures_dir = ensure_output_dirs(config.output_dir)

    print("1. Монте-Карло — базовий сценарій...")
    mc_df      = run_monte_carlo(config, scenario_name="base")
    summary_df = summarize_results(mc_df)

    print("2. Sweep по інтенсивності атак...")
    attack_sweep = run_parameter_sweep(
        config, "attack_intensity", config.attack_sweep_values
    )

    print("3. Sweep по людському чиннику...")
    human_sweep = run_parameter_sweep(
        config, "human_factor", config.human_sweep_values
    )

    print("4. Порівняння всіх сценаріїв...")
    comparison_df = build_scenario_comparison(config)

    # ── Збереження таблиць ───────────────────────────────────────────────────
    print("\nЗберігаємо таблиці...")
    save_csv(mc_df,           tables_dir / "monte_carlo_detailed.csv")
    save_csv(summary_df,      tables_dir / "summary_results.csv")
    save_csv(attack_sweep,    tables_dir / "attack_intensity_sweep.csv")
    save_csv(human_sweep,     tables_dir / "human_factor_sweep.csv")
    save_csv(comparison_df,   tables_dir / "scenario_comparison.csv")

    ranking = summary_df[["method", "integral_score_mean", "integral_score_std", "rank"]]
    save_csv(ranking, tables_dir / "method_ranking.csv")

    # ── Побудова графіків ────────────────────────────────────────────────────
    print("\nБудуємо графіки...")
    plot_integral_scores(summary_df, figures_dir)
    plot_delay_energy(summary_df, figures_dir)
    plot_radar_chart(summary_df, figures_dir)
    plot_scenario_comparison(comparison_df, figures_dir)

    plot_sweep_line(
        attack_sweep, "integral_score_mean", figures_dir,
        "integral_vs_attack_intensity.png",
        "Залежність $\\bar{J}_i$ від інтенсивності атак $\\lambda$",
        "Інтенсивність атак $\\lambda$",
        "Інтегральний показник $\\bar{J}_i$",
    )
    plot_sweep_line(
        human_sweep, "risk_mean", figures_dir,
        "risk_vs_human_factor.png",
        "Залежність ризику від людського чинника $\\gamma$",
        "Людський чинник $\\gamma$",
        "Ризик $\\bar{R}_i$",
    )
    plot_sweep_line(
        attack_sweep, "attack_probability_mean", figures_dir,
        "attack_prob_vs_lambda.png",
        "Ймовірність успішної атаки $P_i$ залежно від $\\lambda$",
        "Інтенсивність атак $\\lambda$",
        "Ймовірність атаки $P_i$",
    )

    print("\n✓ Готово.")
    print(f"  Таблиці : {tables_dir}")
    print(f"  Графіки : {figures_dir}")
    print(f"\nРанжування (базовий сценарій):")
    for _, row in ranking.sort_values("rank").iterrows():
        print(f"  #{int(row['rank'])}  {_label(row['method']):<30}"
              f"  J = {row['integral_score_mean']:.4f} ± {row['integral_score_std']:.4f}")


if __name__ == "__main__":
    main()