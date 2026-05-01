"""Tests for Martensen quadrature weights (eq. 2.9)."""
import numpy as np

from integral_method.quadrature import martensen_weights, nodes


def _exact_log_cos_integral(k: int, t: float) -> float:
    """Exact value of (1/(2π)) ∫_0^{2π} ln(4/e sin²((t-τ)/2)) cos(kτ) dτ.

    Standard Fourier series:  ln(4 sin²((t-τ)/2)) = -2 Σ_{m≥1} cos(m(t-τ))/m.
    Hence ln(4/e sin²(...)) = -1 - 2 Σ_{m≥1} cos(m(t-τ))/m.
    Integral (1/(2π)) ∫ ... cos(kτ) dτ:
        k=0:  -1
        k>0:  -cos(k t)/k
    """
    if k == 0:
        return -1.0
    return -np.cos(k * t) / k


def test_martensen_integrates_low_frequency_exactly():
    M = 16
    R = martensen_weights(M)
    t = nodes(M)
    # Test for a few frequencies k < M (rule has degree 2M-1 of exactness for cos).
    for k in [0, 1, 3, 7, 15]:
        f = np.cos(k * t)              # f(t_j)
        approx = R @ f                 # one approximation per t_i
        exact = np.array([_exact_log_cos_integral(k, ti) for ti in t])
        np.testing.assert_allclose(approx, exact, atol=1e-12)
