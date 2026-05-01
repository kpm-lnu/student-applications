"""Synthetic Cauchy data: harmonic-polynomial ground truth.

We pick a finite set of harmonic basis functions u_n(x,y) = Re(z^n) (z = x + iy) with given
coefficients, evaluate u and ∂u/∂ν on Γ_1, and (for verification) on Γ_0.

This produces a closed-form harmonic field on the whole plane, so reference values on Γ_0 are
exact. Optional Gaussian noise can be added to the Cauchy data on Γ_1 (eq. paper §4 noise model:
percentage of L²-norm).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import Curve


@dataclass
class CauchyData:
    f1: np.ndarray         # u on Γ_1   (2M,)
    f2: np.ndarray         # ∂u/∂ν on Γ_1   (2M,)
    u_ref_inner: np.ndarray      # ground-truth u on Γ_0   (2M,)
    dudn_ref_inner: np.ndarray   # ground-truth ∂u/∂ν on Γ_0   (2M,)
    coeffs: np.ndarray
    noise_pct: float


def _harmonic_value_and_grad(curve: Curve, coeffs: np.ndarray):
    """Evaluate u = Σ a_n Re(z^n) and ∇u along curve points."""
    z = curve.x[:, 0] + 1j * curve.x[:, 1]              # (2M,)
    u = np.zeros(curve.x.shape[0])
    du_dx = np.zeros_like(u)
    du_dy = np.zeros_like(u)
    for n, a in enumerate(coeffs, start=1):
        zn = z ** n
        u += a * zn.real
        # d/dx z^n = n z^{n-1};  d/dy z^n = i n z^{n-1}
        zn_m1 = z ** (n - 1) if n >= 1 else np.zeros_like(z)
        du_dx += a * (n * zn_m1).real
        du_dy += a * (1j * n * zn_m1).real
    return u, du_dx, du_dy


def generate_cauchy_data(
    gamma0: Curve,
    gamma1: Curve,
    coeffs: np.ndarray | None = None,
    noise_pct: float = 0.0,
    seed: int = 0,
) -> CauchyData:
    """Generate harmonic ground-truth Cauchy data on Γ_1 and reference values on Γ_0."""
    if coeffs is None:
        # Default test field: a few low-degree harmonics with assorted signs.
        coeffs = np.array([0.5, -0.3, 0.2, 0.0, -0.1])

    u1, ux1, uy1 = _harmonic_value_and_grad(gamma1, coeffs)
    u0, ux0, uy0 = _harmonic_value_and_grad(gamma0, coeffs)

    f1 = u1.copy()
    f2 = ux1 * gamma1.normal[:, 0] + uy1 * gamma1.normal[:, 1]
    dudn_inner = ux0 * gamma0.normal[:, 0] + uy0 * gamma0.normal[:, 1]

    if noise_pct > 0.0:
        rng = np.random.default_rng(seed)
        # Discrete L²-norm via periodic trapezoid (uniform weight 2π/(2M) cancels in ratio).
        norm_f1 = np.sqrt(np.mean(f1 ** 2))
        norm_f2 = np.sqrt(np.mean(f2 ** 2))
        f1 = f1 + rng.standard_normal(f1.shape) * (noise_pct / 100.0) * norm_f1
        f2 = f2 + rng.standard_normal(f2.shape) * (noise_pct / 100.0) * norm_f2

    return CauchyData(
        f1=f1,
        f2=f2,
        u_ref_inner=u0,
        dudn_ref_inner=dudn_inner,
        coeffs=coeffs,
        noise_pct=noise_pct,
    )
