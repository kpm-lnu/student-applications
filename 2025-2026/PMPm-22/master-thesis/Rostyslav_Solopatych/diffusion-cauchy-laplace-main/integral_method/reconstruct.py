"""Reconstruct u and ∂u/∂ν on Γ_0 (and at interior points) from solved Nyström densities.

§2 Remark, paper p.9:

    u(x_0(t_i))  ≈  Σ_j  [ H̃_{00}(t_i,t_j)/(2M) − ½ R_j(t_i) ] ψ_{0,j}
                    +  Σ_j  H_{10}(t_i,t_j)/(2M)  ψ_{1,j}

    ∂u/∂ν(x_0(t_i))  ≈  −ψ_{0,i} / (2 |x_0'(t_i)|)
                        + 1/(2M) Σ_j K_{00}(t_i,t_j) ψ_{0,j}
                        + 1/(2M) Σ_j K_{10}(t_i,t_j) ψ_{1,j}

(The "H_00" in the paper text uses the smooth-remainder split exactly as H̃_11 is used in eq. 2.11.)
"""
from __future__ import annotations

import numpy as np

from .boundary import Curve
from .kernels import KernelBlocks
from .quadrature import martensen_weights


def reconstruct_on_inner(
    gamma0: Curve,
    blocks: KernelBlocks,
    psi0: np.ndarray,
    psi1: np.ndarray,
    M: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_on_gamma0, dudn_on_gamma0), each (2M,)."""
    R0 = martensen_weights(M)
    inv2M = 1.0 / (2 * M)

    u = (inv2M * blocks.H00_smooth - 0.5 * R0) @ psi0 + inv2M * blocks.H10 @ psi1
    dudn = (
        -psi0 / (2.0 * gamma0.speed)
        + inv2M * blocks.K00 @ psi0
        + inv2M * blocks.K10 @ psi1
    )
    return u, dudn


def evaluate_interior(
    points: np.ndarray,
    gamma0: Curve,
    gamma1: Curve,
    psi0: np.ndarray,
    psi1: np.ndarray,
    M: int,
) -> np.ndarray:
    """
    Evaluate u(x) at arbitrary interior points x ∈ D using the single-layer representation
    (eq. 2.1) discretised by the periodic trapezoid (smooth integrand for x away from Γ0 ∪ Γ1).

    points: (P, 2) array of interior points.
    Returns (P,) values of u.
    """
    inv2M = 1.0 / (2 * M)

    def _layer(xs: np.ndarray, src: Curve, psi: np.ndarray) -> np.ndarray:
        # Single-layer kernel: -1/(2π) ln|x − y| ds(y); ds = |x'| dτ; ψ already absorbs |x'|.
        diff = xs[:, None, :] - src.x[None, :, :]      # (P, 2M, 2)
        r2 = np.sum(diff * diff, axis=-1)
        # H[i, j] = ln(1 / |x_i − y_j|) = -0.5 ln r².  Note ψ_j = μ_j |x'|, dτ = π/M, factor 1/(2π).
        H = -0.5 * np.log(r2)
        return inv2M * (H @ psi)                       # = (1/(2π)) (π/M) Σ ψ_j H_ij

    return _layer(points, gamma0, psi0) + _layer(points, gamma1, psi1)


def evaluate_on_grid(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    fill_mask: np.ndarray,
    gamma0: Curve,
    gamma1: Curve,
    psi0: np.ndarray,
    psi1: np.ndarray,
    M: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Evaluate u on every pixel where ``fill_mask`` is truthy; return a 2D image.

    Pass ``fill_mask = gmask | bmask`` (from ``grid.compute_grid``) to mirror the dataset's
    ``u`` layout: interior pixels + Γ_1 boundary band filled, zeros elsewhere.
    """
    mask_bool = np.asarray(fill_mask, dtype=bool)
    pts = np.column_stack([x_grid[mask_bool], y_grid[mask_bool]])
    u_vals = evaluate_interior(pts, gamma0, gamma1, psi0, psi1, M)
    u_image = np.full(x_grid.shape, fill_value, dtype=float)
    u_image[mask_bool] = u_vals
    return u_image
