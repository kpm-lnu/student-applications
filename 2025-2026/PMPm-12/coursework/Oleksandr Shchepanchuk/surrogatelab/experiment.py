"""Equal-budget surrogate experiment driver."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .problems import Problem
from .sampling import get_sampler
from .surrogates import RBFSurrogate


@dataclass
class ExperimentConfig:
    """Configuration for an equal-budget sampler comparison."""

    n_grid: list[int] = field(default_factory=lambda: [10, 16, 25, 36, 49])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    surrogate_kernel: str = "gaussian"
    nugget: float = 1e-10
    log_transform: bool = True
    test_per_dim: int = 11
    min_distance: float = 0.01

    @property
    def n_max(self) -> int:
        """Largest training budget."""
        return max(self.n_grid)


def make_test_set(
    problem: Problem,
    qoi: str,
    per_dim: int,
    forward=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-factorial unit-cube test grid and QoI values."""
    axes = [np.linspace(0.0, 1.0, per_dim) for _ in range(problem.dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.column_stack([axis_values.ravel() for axis_values in mesh])
    evaluator = forward if forward is not None else problem.forward(qoi)
    return X, evaluator(X)


def run_comparison(
    problem: Problem,
    qois: list[str],
    sampler_names: list[str],
    cfg: ExperimentConfig | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Compare samplers at equal forward-model budgets."""
    cfg = cfg or ExperimentConfig()
    rows: list[dict] = []
    designs: dict = {}

    for qoi in qois:
        forward = problem.forward(qoi)
        X_test, y_test = make_test_set(problem, qoi, cfg.test_per_dim, forward)

        for sampler_name in sampler_names:
            sampler = get_sampler(sampler_name)
            if sampler.is_adaptive:
                sampler.min_distance = cfg.min_distance
                sampler.scoring_log_transform = cfg.log_transform

            for seed in cfg.seeds:
                if sampler.is_adaptive:
                    full_design = sampler.build(
                        cfg.n_max,
                        problem.dim,
                        seed,
                        forward=forward,
                    )
                    full_values = forward(full_design)
                    for n_points in cfg.n_grid:
                        design = full_design[:n_points]
                        values = full_values[:n_points]
                        record = _fit_and_score(design, values, X_test, y_test, cfg)
                        rows.append(
                            {
                                "qoi": qoi,
                                "sampler": sampler_name,
                                "seed": seed,
                                "n": n_points,
                                **record,
                            }
                        )
                        designs[(qoi, sampler_name, seed, n_points)] = design
                else:
                    for n_points in cfg.n_grid:
                        design = sampler.build(n_points, problem.dim, seed)
                        values = forward(design)
                        record = _fit_and_score(design, values, X_test, y_test, cfg)
                        rows.append(
                            {
                                "qoi": qoi,
                                "sampler": sampler_name,
                                "seed": seed,
                                "n": n_points,
                                **record,
                            }
                        )
                        designs[(qoi, sampler_name, seed, n_points)] = design

    return pd.DataFrame(rows), designs


def samples_to_tolerance(
    results: pd.DataFrame,
    tol: float,
    metric: str = "nrmse",
) -> pd.DataFrame:
    """Smallest budget reaching a metric tolerance for each sampler."""
    rows = []
    for (qoi, sampler), group in results.groupby(["qoi", "sampler"]):
        reached = []
        n_censored = 0
        for _seed, seed_group in group.groupby("seed"):
            ordered = seed_group.sort_values("n")
            accepted = ordered[ordered[metric] <= tol]
            if len(accepted):
                reached.append(int(accepted["n"].iloc[0]))
            else:
                n_censored += 1
        if reached:
            values = np.array(reached, dtype=float)
            rows.append(
                {
                    "qoi": qoi,
                    "sampler": sampler,
                    "tol": tol,
                    "n_star_median": float(np.median(values)),
                    "n_star_q25": float(np.percentile(values, 25)),
                    "n_star_q75": float(np.percentile(values, 75)),
                    "n_reached": len(reached),
                    "n_censored": n_censored,
                }
            )
        else:
            rows.append(
                {
                    "qoi": qoi,
                    "sampler": sampler,
                    "tol": tol,
                    "n_star_median": np.nan,
                    "n_star_q25": np.nan,
                    "n_star_q75": np.nan,
                    "n_reached": 0,
                    "n_censored": n_censored,
                }
            )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, metric: str = "nrmse") -> pd.DataFrame:
    """Median and interquartile range over seeds."""
    return (
        results.groupby(["qoi", "sampler", "n"])[metric]
        .agg(
            median="median",
            q25=lambda values: np.nanpercentile(values, 25),
            q75=lambda values: np.nanpercentile(values, 75),
        )
        .reset_index()
    )


def _fit_and_score(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: ExperimentConfig,
) -> dict:
    try:
        surrogate = RBFSurrogate(
            kernel=cfg.surrogate_kernel,
            eps=None,
            nugget=cfg.nugget,
            log_transform=cfg.log_transform,
        ).fit(X, y)
        metrics = compute_metrics(y_test, surrogate.predict(X_test))
        record = metrics.as_dict()
        record.update(
            eps_used=float(getattr(surrogate, "eps_used_", np.nan)),
            cond_kernel=float(surrogate.condition_number),
            failed=False,
        )
    except Exception as exc:
        metric_names = (
            "max_re",
            "i_max_re",
            "max_ae",
            "i_max_ae",
            "mae",
            "mre",
            "mse",
            "rmse",
            "r2",
            "nrmse",
            "eps_used",
            "cond_kernel",
        )
        record = {name: float("nan") for name in metric_names}
        record.update(n_test=len(y_test), failed=True, error=str(exc))
    return record


def _self_test() -> None:
    class ToyProblem(Problem):
        name = "toy"
        param_names = ["a", "b"]
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        qoi_names = ["q"]

        def evaluate(self, mu: np.ndarray) -> dict[str, float]:
            value = 2.0 + np.exp(np.sin(4.0 * mu[0]) + np.cos(4.0 * mu[1]))
            return {"q": float(value)}

    cfg = ExperimentConfig(n_grid=[8, 16, 24], seeds=[0, 1])
    results, _designs = run_comparison(
        ToyProblem(),
        ["q"],
        ["LHS", "Random", "P-greedy", "MEPE"],
        cfg,
        verbose=True,
    )
    assert len(results) == 4 * 2 * 3 and not results["nrmse"].isna().all()
    assert len(samples_to_tolerance(results, tol=0.3)) == 4
    print("[experiment] self-test OK")


if __name__ == "__main__":
    _self_test()
