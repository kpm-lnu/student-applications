"""End-to-end reconstruction tests for the direct integral equation method (§2)."""
import numpy as np
import pytest

from integral_method.boundary import inner_curve, outer_curve
from integral_method.kernels import build_kernel_blocks
from integral_method.reconstruct import reconstruct_on_inner
from integral_method.synthetic import generate_cauchy_data
from integral_method.system import assemble_system
from integral_method.tikhonov import tikhonov_lcurve


def _solve(M: int, noise_pct: float, seed: int = 0):
    g0 = inner_curve(M)
    g1 = outer_curve(M)
    blocks = build_kernel_blocks(g0, g1)
    data = generate_cauchy_data(g0, g1, noise_pct=noise_pct, seed=seed)
    A, b = assemble_system(g1, blocks, data.f1, data.f2, M)
    res = tikhonov_lcurve(A, b)
    psi0 = res.x[: 2 * M]
    psi1 = res.x[2 * M :]
    u, dudn = reconstruct_on_inner(g0, blocks, psi0, psi1, M)
    return u, dudn, data, res


def test_clean_data_accurate():
    u, dudn, data, _ = _solve(M=32, noise_pct=0.0)
    err_u = np.max(np.abs(u - data.u_ref_inner))
    err_dudn = np.max(np.abs(dudn - data.dudn_ref_inner))
    assert err_u < 1e-3, f"u L∞ error {err_u}"
    assert err_dudn < 1e-2, f"∂νu L∞ error {err_dudn}"


def test_ill_posedness_signature():
    """Without regularisation the matrix is severely ill-conditioned (motivates Tikhonov)."""
    from integral_method.kernels import build_kernel_blocks

    M = 32
    g0 = inner_curve(M)
    g1 = outer_curve(M)
    blocks = build_kernel_blocks(g0, g1)
    data = generate_cauchy_data(g0, g1, noise_pct=0.0)
    A, _ = assemble_system(g1, blocks, data.f1, data.f2, M)
    cond = np.linalg.cond(A)
    assert cond > 1e8


def test_3pct_noise_stable():
    u, dudn, data, _ = _solve(M=32, noise_pct=3.0, seed=1)
    err_u = np.max(np.abs(u - data.u_ref_inner))
    err_dudn = np.max(np.abs(dudn - data.dudn_ref_inner))
    # Stability: noise gives a meaningful but bounded error.
    assert err_u < 0.5
    assert err_dudn < 2.0
