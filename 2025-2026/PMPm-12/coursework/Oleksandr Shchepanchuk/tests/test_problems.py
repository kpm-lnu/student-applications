import math

import numpy as np

from surrogatelab.problems import (
    AdvDiffAMICon_FD,
    AdvDiffAMIConProblem,
    BraninProblem,
    HeatAMICon_FD,
    HeatAMIConProblem,
)


def test_heat_amicon_sanity():
    problem = HeatAMIConProblem(n_nodes=51)
    q = problem.evaluate(np.array([1.0, 1.0]))["q"]
    assert abs(q - 1.0 / 12.0) < 1e-3


def test_adv_diff_amicon_diffusion_limit():
    problem = AdvDiffAMIConProblem(n_nodes=51)
    q = problem.evaluate(np.array([1.0, 0.0]))["q"]
    assert abs(q - 1.0 / 12.0) < 1e-3


def test_branin_minima():
    problem = BraninProblem()
    for x_star in ((-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)):
        f_star = problem.evaluate(np.array(x_star))["f"]
        assert abs(f_star - 0.397887) < 1e-3


def test_heat_amicon_monotonic():
    """Higher conductivity in either zone reduces the integral of u."""
    problem = HeatAMIConProblem(n_nodes=51)
    q_low = problem.evaluate(np.array([1.0, 1.0]))["q"]
    q_hi1 = problem.evaluate(np.array([5.0, 1.0]))["q"]
    q_hi2 = problem.evaluate(np.array([1.0, 5.0]))["q"]
    q_hi_both = problem.evaluate(np.array([5.0, 5.0]))["q"]
    assert q_hi1 < q_low
    assert q_hi2 < q_low
    assert q_hi_both < q_hi1
    assert q_hi_both < q_hi2


def test_heat_amicon_fd_sanity():
    """For mu=(1,1) constant kappa: u(x) = x(1-x)/2, integral_0^1 u dx = 1/12."""
    problem = HeatAMICon_FD(n_grid=51)
    q = problem.evaluate(np.array([1.0, 1.0]))["q"]
    assert abs(q - 1.0 / 12.0) < 1e-3, f"got {q}, expected 1/12"


def test_adv_diff_amicon_fd_diffusion_limit():
    """For (nu=1, a=0): u(x) = x(1-x)/2, integral_0^1 u dx = 1/12."""
    problem = AdvDiffAMICon_FD(n_grid=51)
    q = problem.evaluate(np.array([1.0, 0.0]))["q"]
    assert abs(q - 1.0 / 12.0) < 1e-3, f"got {q}, expected 1/12"


def test_amicon_fd_matches_fem_solution():
    """FD and P1-FEM discretizations must agree to within ~1% on the same PDE."""
    for mu in (np.array([0.5, 9.0]), np.array([2.0, 0.3]), np.array([1.0, 1.0])):
        q_fd = HeatAMICon_FD(n_grid=51).evaluate(mu)["q"]
        q_fem = HeatAMIConProblem(n_nodes=51).evaluate(mu)["q"]
        assert abs(q_fd - q_fem) / max(abs(q_fem), 1e-12) < 0.02, (
            f"Heat mu={mu}: FD={q_fd}, FEM={q_fem}"
        )
    for mu in (np.array([0.1, 2.0]), np.array([0.01, 5.0]), np.array([1.0, 0.0])):
        q_fd = AdvDiffAMICon_FD(n_grid=51).evaluate(mu)["q"]
        q_fem = AdvDiffAMIConProblem(n_nodes=51).evaluate(mu)["q"]
        assert abs(q_fd - q_fem) / max(abs(q_fem), 1e-12) < 0.05, (
            f"AdvDiff mu={mu}: FD={q_fd}, FEM={q_fem}"
        )
