import numpy as np

from surrogatelab import fem
from surrogatelab.problems import HeatProblem, ReactionDiffusionProblem


def test_lumped_fem_equals_old_fdm():
    n_nodes = 21
    dx = 0.2
    rng = np.random.default_rng(8)
    y = rng.normal(size=n_nodes)

    reference = np.empty_like(y)
    reference[1:-1] = (y[:-2] - 2.0 * y[1:-1] + y[2:]) / dx**2
    reference[0] = 2.0 * (y[1] - y[0]) / dx**2
    reference[-1] = 2.0 * (y[-2] - y[-1]) / dx**2

    actual = fem.laplacian_neumann(n_nodes, dx, mass="lumped") @ y
    np.testing.assert_allclose(actual, reference, atol=1e-12)


def test_consistent_fem_changes_qoi_slightly():
    lumped = ReactionDiffusionProblem(pde_nx=81, mass="lumped")
    consistent = ReactionDiffusionProblem(pde_nx=81, mass="consistent")
    base_mu = lumped.p_base[list(lumped.active)]

    qoi_lumped = lumped.evaluate(base_mu)
    qoi_consistent = consistent.evaluate(base_mu)

    for qoi in ("J2",):
        relative_change = abs(qoi_consistent[qoi] - qoi_lumped[qoi]) / abs(qoi_lumped[qoi])
        assert 0.001 < relative_change < 0.05


def test_heat_problem_known_solution():
    problem = HeatProblem(n_nodes=21)
    q = problem.evaluate(np.array([1.0, 1.0]))["q"]
    assert abs(q - 1.0 / 12.0) < 1e-3


def test_heat_piecewise_symmetry():
    problem = HeatProblem(n_nodes=21)
    q = problem.evaluate(np.array([2.0, 2.0]))["q"]
    assert abs(q - 1.0 / 24.0) < 1e-3
