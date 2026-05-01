"""Discretised Nyström system (eq. 2.11) for the Cauchy problem on a doubly connected domain.

Recall (2.11):

    Row block 1 (Dirichlet match on Γ_1):
        1/(2M) Σ_j ψ0_j H_{01}(t_i,t_j) + Σ_j ψ1_j [ H̃_{11}(t_i,t_j)/(2M) − ½ R_j(t_i) ] = f1_i

    Row block 2 (Neumann match on Γ_1):
        1/(2M) Σ_j ψ0_j K_{01}(t_i,t_j) + 1/(2M) Σ_j ψ1_j K_{11}(t_i,t_j)
            + ψ1_i / (2 |x_1'(t_i)|) = f2_i

with f1_i = f1(x_1(t_i)), f2_i = f2(x_1(t_i)).
"""
from __future__ import annotations

import numpy as np

from .boundary import Curve
from .kernels import KernelBlocks
from .quadrature import martensen_weights


def assemble_system(
    gamma1: Curve,
    blocks: KernelBlocks,
    f1: np.ndarray,
    f2: np.ndarray,
    M: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, b) of shapes ((4M, 4M), (4M,))."""
    R = martensen_weights(M)                              # (2M, 2M)
    inv2M = 1.0 / (2 * M)

    A = np.zeros((4 * M, 4 * M), dtype=float)
    # Top-left:  1/(2M) H01
    A[: 2 * M, : 2 * M] = inv2M * blocks.H01
    # Top-right: H̃11/(2M) − 0.5 R
    A[: 2 * M, 2 * M :] = inv2M * blocks.H11_smooth - 0.5 * R
    # Bottom-left: 1/(2M) K01
    A[2 * M :, : 2 * M] = inv2M * blocks.K01
    # Bottom-right: 1/(2M) K11 + diag(1/(2|x1'|))
    A[2 * M :, 2 * M :] = inv2M * blocks.K11
    diag_jump = 1.0 / (2.0 * gamma1.speed)
    A[2 * M :, 2 * M :] += np.diag(diag_jump)

    b = np.concatenate([f1, f2])
    return A, b
