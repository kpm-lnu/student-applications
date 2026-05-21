import numpy as np
from scipy.stats import qmc

from surrogatelab.sampling import (
    BetaGreedySampler,
    FGreedySampler,
    GreedyContext,
    PGreedySampler,
)
from surrogatelab.surrogates import RBFSurrogate


def _target(design: np.ndarray) -> np.ndarray:
    return 2.0 + np.sin(np.pi * design[:, 0]) * np.cos(np.pi * design[:, 1])


def _context() -> GreedyContext:
    engine = qmc.LatinHypercube(d=2, scramble=True, rng=3)
    design = engine.random(5)
    values = _target(design)
    near = design + 5e-4
    rng = np.random.default_rng(11)
    pool = np.vstack([near, rng.uniform(0.0, 1.0, (40, 2))])
    pool = np.clip(pool, 0.0, 1.0)
    surrogate = RBFSurrogate(
        kernel="gaussian",
        eps=1.0,
        log_transform=False,
    ).fit(design, values)
    return GreedyContext(
        pool=pool,
        available=np.ones(pool.shape[0], dtype=bool),
        X=design,
        y=values,
        surrogate=surrogate,
        iteration=design.shape[0],
        state={},
    )


def _argmax_set(scores: np.ndarray) -> set[int]:
    maximum = np.nanmax(scores)
    return set(np.flatnonzero(np.isclose(scores, maximum, rtol=1e-12, atol=1e-12)))


def test_beta_greedy_reduces_to_p_greedy():
    context = _context()
    beta_scores = BetaGreedySampler(beta=0.0).score(context)
    p_scores = PGreedySampler().score(context)
    assert _argmax_set(beta_scores) == _argmax_set(p_scores)


def test_beta_greedy_reduces_to_f_greedy():
    context = _context()
    beta_scores = BetaGreedySampler(beta=1.0).score(context)
    f_scores = FGreedySampler().score(context)
    assert _argmax_set(beta_scores) == _argmax_set(f_scores)


def test_beta_greedy_finite_scores():
    scores = BetaGreedySampler(beta=0.5).score(_context())
    assert np.all(np.isfinite(scores))
