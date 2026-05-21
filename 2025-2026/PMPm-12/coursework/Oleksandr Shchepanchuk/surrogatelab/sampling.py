"""Design-of-experiments samplers on the unit cube.

Two families share the :class:`Sampler` interface:
  * Space-filling (one-shot, non-adaptive):
        :class:`RandomSampler`, :class:`LHSSampler`, :class:`HaltonSampler`.
  * Greedy adaptive (extend an existing design one point at a time,
    refitting the surrogate at each step):
        :class:`PGreedySampler`  - max kernel power function
        :class:`FGreedySampler`  - max surrogate-residual magnitude
        :class:`BetaGreedySampler` (beta in [0, 1]) - convex blend of
            P-greedy and f-greedy criteria
        :class:`MEPESampler`     - LOOCV-weighted leave-one-out residual
        :class:`EIGFSampler`     - expected improvement for global fit.

Samplers self-register into :data:`SAMPLER_REGISTRY` via
`@register_sampler`; instantiate by name with :func:`get_sampler`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from .surrogates import RBFSurrogate, pairwise_distances


SAMPLER_REGISTRY: dict[str, type] = {}


def register_sampler(cls: type) -> type:
    """Register a sampler class by its name."""
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must define a non-empty 'name'")
    SAMPLER_REGISTRY[cls.name] = cls
    return cls


def get_sampler(name: str, **overrides) -> "Sampler":
    """Instantiate a registered sampler."""
    if name not in SAMPLER_REGISTRY and name.startswith("β-greedy"):
        sampler = BetaGreedySampler(beta=_parse_beta(name, overrides.pop("beta", 0.5)))
        for key, value in overrides.items():
            setattr(sampler, key, value)
        return sampler
    if name not in SAMPLER_REGISTRY:
        raise KeyError(f"unknown sampler '{name}'. Available: {sorted(SAMPLER_REGISTRY)}")
    sampler_cls = SAMPLER_REGISTRY[name]
    if sampler_cls.__name__ == "BetaGreedySampler" and "beta" in overrides:
        sampler = sampler_cls(beta=overrides.pop("beta"))
    else:
        sampler = sampler_cls()
    for key, value in overrides.items():
        setattr(sampler, key, value)
    return sampler


def list_samplers() -> dict[str, bool]:
    """Return sampler names mapped to their adaptive flag."""
    return {name: cls.is_adaptive for name, cls in sorted(SAMPLER_REGISTRY.items())}


class Sampler(ABC):
    """Abstract design sampler on the unit cube."""

    name: str = "abstract"
    is_adaptive: bool = False

    @abstractmethod
    def build(
        self,
        n_points: int,
        dim: int,
        seed: int,
        *,
        forward: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return an (n_points, dim) design in [0, 1]^d."""

    def __repr__(self) -> str:
        kind = "adaptive" if self.is_adaptive else "space-filling"
        return f"<Sampler {self.name!r} ({kind})>"


class SpaceFillingSampler(Sampler):
    """Base class for static, non-adaptive designs."""

    is_adaptive = False

    def build(
        self,
        n_points: int,
        dim: int,
        seed: int,
        *,
        forward: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Draw a fresh space-filling design."""
        return np.atleast_2d(self._design(n_points, dim, seed))

    @abstractmethod
    def _design(self, n_points: int, dim: int, seed: int) -> np.ndarray:
        ...


@register_sampler
class RandomSampler(SpaceFillingSampler):
    """Uniform independent random points."""

    name = "Random"

    def _design(self, n_points: int, dim: int, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).uniform(0.0, 1.0, size=(n_points, dim))


@register_sampler
class LHSSampler(SpaceFillingSampler):
    """Latin hypercube design."""

    name = "LHS"

    def _design(self, n_points: int, dim: int, seed: int) -> np.ndarray:
        engine = qmc.LatinHypercube(
            d=dim,
            scramble=True,
            optimization="random-cd",
            rng=seed,
        )
        return engine.random(n_points)


@register_sampler
class HaltonSampler(SpaceFillingSampler):
    """Scrambled Halton sequence."""

    name = "Halton"

    def _design(self, n_points: int, dim: int, seed: int) -> np.ndarray:
        return qmc.Halton(d=dim, scramble=True, rng=seed).random(n_points)


@dataclass
class GreedyContext:
    """State passed to an adaptive acquisition rule."""

    pool: np.ndarray
    available: np.ndarray
    X: np.ndarray
    y: np.ndarray
    surrogate: RBFSurrogate
    iteration: int
    state: dict


class GreedySampler(Sampler):
    """Base class for sequential greedy samplers."""

    is_adaptive = True
    pool_size: int = 4000
    pool_seed: int = 999
    initial_size: int = 5
    scoring_kernel: str = "gaussian"
    scoring_eps: float = 1.0
    nugget: float = 1e-10
    scoring_log_transform: bool = True
    min_distance: float = 0.01

    def build(
        self,
        n_points: int,
        dim: int,
        seed: int,
        *,
        forward: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Build a nested adaptive design."""
        if forward is None:
            raise ValueError(f"adaptive sampler '{self.name}' needs a forward callable")
        if n_points < self.initial_size:
            raise ValueError("n_points below initial design size")

        pool = self._make_pool(dim)
        design = self._snap_to_pool(self._initial_design(dim, seed), pool)
        values = np.asarray(forward(design), dtype=float).ravel()
        available = ~_membership_mask(pool, design)
        self._block_neighbours(pool, design, available)
        state: dict = {}

        while design.shape[0] < n_points:
            surrogate = RBFSurrogate(
                kernel=self.scoring_kernel,
                eps=self.scoring_eps,
                nugget=self.nugget,
                log_transform=self.scoring_log_transform,
            ).fit(design, values)
            context = GreedyContext(
                pool=pool,
                available=available,
                X=design,
                y=values,
                surrogate=surrogate,
                iteration=design.shape[0],
                state=state,
            )
            scores = np.asarray(self.score(context), dtype=float).ravel()
            scores = np.where(available, scores, -np.inf)
            if not np.any(np.isfinite(scores)):
                break

            selected = int(np.argmax(scores))
            new_point = pool[selected:selected + 1]
            new_value = np.asarray(forward(new_point), dtype=float).ravel()
            self.after_select(context, selected, float(new_value[0]))

            design = np.vstack([design, new_point])
            values = np.append(values, new_value)
            available[selected] = False
            self._block_neighbours(pool, new_point, available)

        return design

    @abstractmethod
    def score(self, context: GreedyContext) -> np.ndarray:
        """Acquisition score for every pool candidate."""

    def after_select(
        self,
        context: GreedyContext,
        selected: int,
        y_new: float,
    ) -> None:
        """Hook called after an adaptive point is selected."""
        return None

    def _block_neighbours(
        self,
        pool: np.ndarray,
        points: np.ndarray,
        available: np.ndarray,
    ) -> None:
        if self.min_distance <= 0.0:
            return
        distances = pairwise_distances(pool, np.atleast_2d(points))
        available[distances.min(axis=1) < self.min_distance] = False

    def _make_pool(self, dim: int) -> np.ndarray:
        rng = np.random.default_rng(self.pool_seed)
        return rng.uniform(0.0, 1.0, size=(self.pool_size, dim))

    def _initial_design(self, dim: int, seed: int) -> np.ndarray:
        engine = qmc.LatinHypercube(
            d=dim,
            scramble=True,
            optimization="random-cd",
            rng=seed,
        )
        return engine.random(self.initial_size)

    @staticmethod
    def _snap_to_pool(design: np.ndarray, pool: np.ndarray) -> np.ndarray:
        distances = pairwise_distances(design, pool)
        return pool[np.argmin(distances, axis=1)]


@register_sampler
class PGreedySampler(GreedySampler):
    """P-greedy acquisition based on the kernel power function."""

    name = "P-greedy"

    def score(self, context: GreedyContext) -> np.ndarray:
        """Score candidates by squared power function."""
        power = context.surrogate.power_function(context.pool)
        return power * power


@register_sampler
class FGreedySampler(GreedySampler):
    """f-greedy acquisition based on leave-one-out residuals."""

    name = "f-greedy"

    def score(self, context: GreedyContext) -> np.ndarray:
        """Score candidates by nearest-neighbour squared LOO residual."""
        residuals = context.surrogate.loo_residuals()
        nearest = _nearest_index(context.pool, context.X)
        return (residuals * residuals)[nearest]


@register_sampler
class BetaGreedySampler(GreedySampler):
    """β-greedy family from Wenzel-Santin-Haasdonk 2023.

    Score: |r_n(x)|^β * P_Xn(x)^(1-β), β in [0, 1].
    β=0 gives P-greedy, β=1 gives f-greedy, and β=0.5 balances both
    terms without dividing by the power function.
    """

    name = "β-greedy(β=0.5)"

    def __init__(self, beta: float = 0.5, **kwargs):
        super().__init__()
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.beta = float(beta)
        self.name = f"β-greedy(β={beta:g})"
        for key, value in kwargs.items():
            setattr(self, key, value)

    def score(self, context: GreedyContext) -> np.ndarray:
        """Rank candidates by log β-greedy score."""
        residuals = context.surrogate.loo_residuals()
        nearest = _nearest_index(context.pool, context.X)
        residual_abs = np.abs(residuals)[nearest]
        power = context.surrogate.power_function(context.pool)
        eps_log = 1e-300
        return (
            self.beta * np.log(residual_abs + eps_log)
            + (1.0 - self.beta) * np.log(power + eps_log)
        )


@register_sampler
class MEPESampler(GreedySampler):
    """Maximum expected prediction error acquisition."""

    name = "MEPE"

    def score(self, context: GreedyContext) -> np.ndarray:
        """Blend nearest-neighbour LOO residuals with the power function."""
        state = context.state
        if "pending" in state:
            true_error_sq, loo_error_sq = state.pop("pending")
            if loo_error_sq > 1e-300:
                state["alpha"] = 0.99 * min(0.5 * true_error_sq / loo_error_sq, 1.0)
        alpha = state.setdefault("alpha", 0.5)

        residuals = context.surrogate.loo_residuals()
        nearest = _nearest_index(context.pool, context.X)
        loo_term = (residuals * residuals)[nearest]
        power = context.surrogate.power_function(context.pool)
        power_term = power * power

        state["_loo_term"] = loo_term
        return alpha * loo_term + (1.0 - alpha) * power_term

    def after_select(
        self,
        context: GreedyContext,
        selected: int,
        y_new: float,
    ) -> None:
        """Store the MEPE alpha update for the next iteration."""
        surrogate = context.surrogate
        new_point = context.pool[selected:selected + 1]
        predicted = float(surrogate.latent(new_point)[0])
        observed = float(surrogate.standardize_target(np.array([y_new]))[0])
        true_error_sq = (predicted - observed) ** 2
        loo_error_sq = float(context.state["_loo_term"][selected])
        context.state["pending"] = (true_error_sq, loo_error_sq)


@register_sampler
class EIGFSampler(GreedySampler):
    """Expected improvement for global fit acquisition."""

    name = "EIGF"

    def score(self, context: GreedyContext) -> np.ndarray:
        """Blend nearest-observation discrepancy with predictive variance."""
        prediction = context.surrogate.latent(context.pool)
        target = context.surrogate.standardize_target(context.y)
        nearest = _nearest_index(context.pool, context.X)
        exploitation = (prediction - target[nearest]) ** 2
        power = context.surrogate.power_function(context.pool)
        return exploitation + power * power


def _nearest_index(pool: np.ndarray, design: np.ndarray) -> np.ndarray:
    return np.argmin(pairwise_distances(pool, design), axis=1)


def _parse_beta(name: str, default: float) -> float:
    marker = "β="
    if marker not in name:
        return float(default)
    tail = name.split(marker, 1)[1].rstrip(")")
    return float(tail)


def _membership_mask(
    pool: np.ndarray,
    design: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    if design.size == 0:
        return np.zeros(pool.shape[0], dtype=bool)
    return pairwise_distances(pool, design).min(axis=1) < tol


def _self_test() -> None:
    print("[sampling] registered:", list_samplers())

    def forward(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return 2.0 + np.exp(np.sin(3.0 * X[:, 0]) + np.cos(3.0 * X[:, 1]))

    for name in list_samplers():
        sampler = get_sampler(name)
        design = sampler.build(24, 2, seed=1, forward=forward)
        assert design.shape == (24, 2)
        assert design.min() >= -1e-9 and design.max() <= 1.0 + 1e-9
        assert len(np.unique(np.round(design, 9), axis=0)) == 24, name
        print(f"  {name:12s} -> design {design.shape}, all unique, inside cube")
    print("[sampling] self-test OK")


if __name__ == "__main__":
    _self_test()
