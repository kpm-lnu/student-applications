import numpy as np

from surrogatelab.sampling import get_sampler, list_samplers


def positive_forward(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X)
    return 2.0 + np.exp(np.sin(3.0 * X[:, 0]) + np.cos(3.0 * X[:, 1]))


def assert_valid_design(design: np.ndarray) -> None:
    assert design.shape == (20, 2)
    assert design.min() >= -1e-12
    assert design.max() <= 1.0 + 1e-12
    assert len(np.unique(np.round(design, 12), axis=0)) == design.shape[0]


def test_production_samplers_registered():
    samplers = list_samplers()
    assert "EIGF" in samplers
    assert "MEPE" in samplers
    assert "β-greedy(β=0.5)" in samplers
    assert "f/P-greedy" not in samplers


def test_eigf_builds_valid_design():
    sampler = get_sampler("EIGF")
    design = sampler.build(20, 2, seed=3, forward=positive_forward)
    assert_valid_design(design)
