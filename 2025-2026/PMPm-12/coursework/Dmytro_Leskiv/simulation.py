from random import Random
from typing import Dict, List, Optional

import pandas as pd

from config import BASE_THREATS, ModelConfig, SCENARIOS
from core import (
    EnvironmentParams,
    build_threats,
    clamp,
    compute_integral_scores,
    evaluate_method,
    get_default_methods,
)


def build_environment(
    scenario_name: str,
    rng: Random,
    override_parameter: Optional[str] = None,
    override_value: Optional[float] = None,
) -> EnvironmentParams:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Невідомий сценарій: {scenario_name}")

    base = dict(SCENARIOS[scenario_name])
    if override_parameter is not None and override_value is not None:
        base[override_parameter] = override_value

    return EnvironmentParams(
        attack_intensity   = clamp(rng.gauss(base["attack_intensity"],   0.03), 0.05, 1.20),
        human_factor       = clamp(rng.gauss(base["human_factor"],       0.03), 0.01, 1.00),
        channel_instability= clamp(rng.gauss(base["channel_instability"],0.02), 0.01, 1.00),
        node_count         = max(2, int(round(rng.gauss(base["node_count"], 0.6)))),
        data_volume        = max(1.0, rng.gauss(base["data_volume"], 2.0)),
    )


def run_single_experiment(
    config: ModelConfig,
    scenario_name: str = "base",
    rng: Optional[Random] = None,
    override_parameter: Optional[str] = None,
    override_value: Optional[float] = None,
) -> pd.DataFrame:
    # ВИПРАВЛЕНО: передаємо rng ззовні; якщо None — створюємо,
    # але це означає фіксований результат — використовувати тільки для дебагу.
    local_rng = rng if rng is not None else Random(config.seed)

    env     = build_environment(scenario_name, local_rng, override_parameter, override_value)
    methods = get_default_methods()
    threats = build_threats(BASE_THREATS)

    raw: List[Dict] = []
    for method in methods:
        res = evaluate_method(method, env, threats, local_rng)
        res["scenario"]                  = scenario_name
        res["attack_intensity_input"]    = env.attack_intensity
        res["human_factor_input"]        = env.human_factor
        res["channel_instability_input"] = env.channel_instability
        res["node_count_input"]          = env.node_count
        res["data_volume_input"]         = env.data_volume
        raw.append(res)

    return pd.DataFrame(compute_integral_scores(raw, config.weights))


def run_monte_carlo(
    config: ModelConfig,
    scenario_name: str = "base",
) -> pd.DataFrame:
    rng    = Random(config.seed)
    frames = []
    for run_id in range(config.monte_carlo_runs):
        df          = run_single_experiment(config, scenario_name, rng=rng)
        df["run_id"] = run_id + 1
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run_all_scenarios(config: ModelConfig) -> pd.DataFrame:
    """Запуск Монте-Карло для всіх сценаріїв і об'єднання результатів."""
    frames = []
    for sc_name in SCENARIOS:
        df = run_monte_carlo(config, scenario_name=sc_name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "security_score", "post_quantum_score", "risk", "attack_probability",
        "delay", "computational_cost", "energy_cost", "human_factor_score",
        "overhead", "integral_score",
        "security_score_norm", "post_quantum_score_norm", "risk_norm",
        "delay_norm", "computational_cost_norm", "energy_cost_norm",
        "human_factor_score_norm", "overhead_norm",
    ]

    group_cols = ["scenario", "method"] if "scenario" in results_df.columns else ["method"]

    summary = results_df.groupby(group_cols)[numeric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        c[0] if c[1] == "" else f"{c[0]}_{c[1]}"
        for c in summary.columns.to_flat_index()
    ]
    summary = summary.sort_values(
        by=["scenario", "integral_score_mean"] if "scenario" in summary.columns
        else ["integral_score_mean"],
        ascending=[True, False] if "scenario" in summary.columns else [False],
    ).reset_index(drop=True)

    if "scenario" in summary.columns:
        summary["rank"] = (
            summary.groupby("scenario")["integral_score_mean"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
    else:
        summary["rank"] = range(1, len(summary) + 1)

    return summary


def build_scenario_comparison(config: ModelConfig) -> pd.DataFrame:
    """
    Зведена таблиця: для кожного сценарію — середній J_i та ранг методів.
    Зручно для таблиці у розділі 3.
    """
    rows = []
    for sc_name in SCENARIOS:
        df      = run_monte_carlo(config, scenario_name=sc_name)
        summary = summarize_results(df)
        for _, row in summary.iterrows():
            rows.append({
                "scenario":           sc_name,
                "method":             row["method"],
                "integral_score_mean":row["integral_score_mean"],
                "integral_score_std": row["integral_score_std"],
                "risk_mean":          row["risk_mean"],
                "attack_probability_mean": row["attack_probability_mean"],
                "rank":               row["rank"],
            })
    return pd.DataFrame(rows)


def run_parameter_sweep(
    config: ModelConfig,
    parameter_name: str,
    values: List[float],
    scenario_name: str = "base",
) -> pd.DataFrame:
    rng     = Random(config.seed + 1000)
    all_rows = []

    for val in values:
        frames = []
        for _ in range(config.monte_carlo_runs):
            df = run_single_experiment(
                config, scenario_name, rng=rng,
                override_parameter=parameter_name,
                override_value=val,
            )
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        grouped  = (
            combined.groupby("method", as_index=False)[
                ["integral_score", "risk", "delay",
                 "energy_cost", "attack_probability"]
            ].agg(["mean", "std"])
        )
        grouped.columns = [
            c[0] if c[1] == "" else f"{c[0]}_{c[1]}"
            for c in grouped.columns.to_flat_index()
        ]
        grouped["sweep_parameter"] = parameter_name
        grouped["sweep_value"]     = val
        all_rows.append(grouped)

    return pd.concat(all_rows, ignore_index=True)