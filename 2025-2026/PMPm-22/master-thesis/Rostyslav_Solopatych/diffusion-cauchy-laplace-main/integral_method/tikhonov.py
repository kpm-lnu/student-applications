"""
Tikhonov regularisation with L-curve parameter choice.

We solve

    min_x ‖A x − b‖² + λ ‖x‖²

via the thin SVD `A = U Σ V^T`.  Filter factors `f_i(λ) = σ_i² / (σ_i² + λ)` give

    x_λ = V diag(f_i / σ_i) U^T b
    ‖A x_λ − b‖² = Σ_i ((1 − f_i) (U^T b)_i)²  + ‖b - U U^T b‖²    (last term is data outside range)
    ‖x_λ‖²     = Σ_i (f_i / σ_i)² (U^T b)_i²

The L-curve corner is selected as the point of maximum curvature on the log-log curve
(ρ(λ) = log‖A x_λ − b‖, η(λ) = log‖x_λ‖) — Hansen's standard heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TikhonovResult:
    x: np.ndarray            # regularised solution at chosen λ
    lam: float               # selected regularisation parameter
    lambdas: np.ndarray      # full λ grid
    residual_norms: np.ndarray   # ‖A x_λ − b‖₂ per λ
    solution_norms: np.ndarray   # ‖x_λ‖₂ per λ
    corner_index: int


def _lcurve_corner(rho: np.ndarray, eta: np.ndarray) -> int:
    """Index of maximum curvature on the log-log L-curve (Hansen)."""
    log_rho = np.log(rho)
    log_eta = np.log(eta)
    # First and second derivatives w.r.t. parameter index (uniform spacing on log-λ).
    drho = np.gradient(log_rho)
    deta = np.gradient(log_eta)
    ddrho = np.gradient(drho)
    ddeta = np.gradient(deta)
    curvature = (drho * ddeta - ddrho * deta) / (drho ** 2 + deta ** 2) ** 1.5
    # Avoid endpoints (numerically poor gradient there).
    interior = np.arange(2, len(curvature) - 2)
    return int(interior[np.argmax(curvature[interior])])


def tikhonov_lcurve(
    A: np.ndarray,
    b: np.ndarray,
    lam_min: float = 1e-14,
    lam_max: float = 1e2,
    n_lambda: int = 200,
) -> TikhonovResult:
    """Solve A x ≈ b via Tikhonov, selecting λ by the L-curve corner."""
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
    Utb = U.T @ b                                # (k,)
    # Out-of-range residual: contributes a constant to ‖A x_λ − b‖.
    out_of_range_sq = float(np.linalg.norm(b) ** 2 - np.sum(Utb ** 2))
    out_of_range_sq = max(out_of_range_sq, 0.0)

    lambdas = np.geomspace(lam_min, lam_max, n_lambda)
    sigma2 = sigma ** 2

    # x_λ = V diag(σ_i / (σ_i² + λ)) U^T b — numerically safe even for σ_i → 0.
    coef = sigma[None, :] * Utb[None, :] / (sigma2[None, :] + lambdas[:, None])
    sol_norm_sq = np.sum(coef ** 2, axis=1)
    # filter factors only used for residual; (1-f) Utb_i = λ Utb_i / (σ_i² + λ)
    one_minus_f = lambdas[:, None] / (sigma2[None, :] + lambdas[:, None])
    res_norm_sq = np.sum((one_minus_f * Utb[None, :]) ** 2, axis=1) + out_of_range_sq
    sol_norms = np.sqrt(sol_norm_sq)
    res_norms = np.sqrt(np.maximum(res_norm_sq, 0.0))

    # Guard against zeros that would break the log.
    eps = np.finfo(float).tiny
    res_for_log = np.maximum(res_norms, eps)
    sol_for_log = np.maximum(sol_norms, eps)
    idx = _lcurve_corner(res_for_log, sol_for_log)
    lam = float(lambdas[idx])
    x = Vt.T @ coef[idx]
    return TikhonovResult(
        x=x,
        lam=lam,
        lambdas=lambdas,
        residual_norms=res_norms,
        solution_norms=sol_norms,
        corner_index=idx,
    )
