"""
Trigonometric quadrature for the periodic log-singular kernel of eq. (2.9).

We need the rule

    1/(2π) ∫_0^{2π} f(τ) ln(4/e · sin²((t−τ)/2)) dτ  ≈  Σ_{k=0}^{2M-1} R_k(t) f(t_k)

at the equidistant nodes t_j = j·π/M.  The classical Martensen / Kress weights (Kress, *Linear
Integral Equations*, §12) are

    R_k(t) = -1/(2M) - (1/M) Σ_{m=1}^{M-1} (1/m) cos(m (t - t_k))  -  cos(M (t - t_k)) / (2 M²).

This is the rule appropriate for the kernel ln(4/e sin²((t-τ)/2)); it integrates trigonometric
polynomials of degree < 2M exactly.
"""
from __future__ import annotations

import numpy as np


def nodes(M: int) -> np.ndarray:
    """Equidistant nodes t_j = j π / M, j = 0..2M-1."""
    return np.arange(2 * M) * np.pi / M


def martensen_weights(M: int) -> np.ndarray:
    """
    Return R[i, k] = R_k(t_i), the (2M, 2M) Kress weight matrix.

    R_k(t_i) depends only on the difference t_i - t_k = (i-k)π/M, so the matrix is circulant.
    """
    t = nodes(M)
    diff = t[:, None] - t[None, :]            # (2M, 2M)
    out = -1.0 / (2 * M) * np.ones_like(diff)
    if M > 1:
        m = np.arange(1, M)                   # (M-1,)
        # broadcast: (2M, 2M, M-1)
        cos_terms = np.cos(m[None, None, :] * diff[..., None])
        out -= (1.0 / M) * np.sum(cos_terms / m[None, None, :], axis=-1)
    out -= np.cos(M * diff) / (2.0 * M * M)
    return out
