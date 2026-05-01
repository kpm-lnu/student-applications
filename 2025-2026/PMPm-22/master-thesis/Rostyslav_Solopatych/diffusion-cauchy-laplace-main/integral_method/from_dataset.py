"""Bridge between the pixel-grid dataset format and the §2 integral-method pipeline.

Dataset samples store Dirichlet/Neumann data on **boundary pixels** of Γ_1 (the outer ellipse).
The integral method needs them on **parametric nodes** ``t_j = j π / M``. We resample by mapping
each boundary pixel to its ellipse parameter via ``t = arctan2(y, x / 1.3) mod 2π`` and then
periodic 1-D linear interpolation onto the uniform parametric grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import Curve, inner_curve, outer_curve
from .kernels import build_kernel_blocks
from .reconstruct import evaluate_on_grid
from .system import assemble_system
from .tikhonov import tikhonov_lcurve


# Ellipse parameters of Γ_1 (must match dataset/generate_harmonic_dataset.py and boundary.outer_curve).
_GAMMA1_A = 1.3
_GAMMA1_B = 1.0
_SQUARE_BOUNDS = (-1.5, 1.5, -1.5, 1.5)


@dataclass
class IntegralReconstruction:
    u_image: np.ndarray         # (H, W) reconstructed u, zero outside the fill mask
    fill_mask: np.ndarray       # (H, W) bool — pixels actually evaluated
    f1: np.ndarray              # (2M,) Dirichlet on parametric nodes (resampled)
    f2: np.ndarray              # (2M,) Neumann on parametric nodes (resampled)
    psi0: np.ndarray            # (2M,) inner density
    psi1: np.ndarray            # (2M,) outer density
    lam: float
    cond_A: float


def _pixel_grid(pixel_res: int) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = _SQUARE_BOUNDS
    x_edges = np.linspace(x_min, x_max, pixel_res + 1)
    y_edges = np.linspace(y_min, y_max, pixel_res + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return np.meshgrid(x_centers, y_centers, indexing="xy")


def cauchy_from_boundary_pixels(
    dirichlet: np.ndarray,
    neumann: np.ndarray,
    bmask: np.ndarray,
    M: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample boundary-pixel Cauchy data onto parametric nodes ``t_j = j π / M``.

    Parameters
    ----------
    dirichlet, neumann : (H, W) arrays — values on boundary pixels, zero elsewhere.
    bmask : (H, W) integer/bool array, 1 on Γ_1 boundary pixels.
    M : half number of parametric nodes; output length is 2M.
    """
    pixel_res = bmask.shape[0]
    if bmask.shape[1] != pixel_res:
        raise ValueError("bmask must be square")
    x_grid, y_grid = _pixel_grid(pixel_res)
    
    boundary_mask_filter = bmask.astype(bool)
    
    x_b = x_grid[boundary_mask_filter]
    y_b = y_grid[boundary_mask_filter]
    d_b = dirichlet[boundary_mask_filter]
    n_b = neumann[boundary_mask_filter]

    # Map ellipse-pixel to its parameter t ∈ [0, 2π). Γ_1: x = 1.3 cos t, y = sin t.
    t_b = np.mod(np.arctan2(y_b / _GAMMA1_B, x_b / _GAMMA1_A), 2 * np.pi)
    order = np.argsort(t_b)
    t_sorted = t_b[order]
    d_sorted = d_b[order]
    n_sorted = n_b[order]

    # Periodic extension for np.interp (no native period support).
    t_ext = np.concatenate([t_sorted - 2 * np.pi, t_sorted, t_sorted + 2 * np.pi])
    d_ext = np.concatenate([d_sorted, d_sorted, d_sorted])
    n_ext = np.concatenate([n_sorted, n_sorted, n_sorted])

    t_nodes = np.arange(2 * M) * np.pi / M
    f1 = np.interp(t_nodes, t_ext, d_ext)
    f2 = np.interp(t_nodes, t_ext, n_ext)
    return f1, f2


def reconstruct_from_dataset_sample(
    dirichlet: np.ndarray,
    neumann: np.ndarray,
    gmask: np.ndarray,
    bmask: np.ndarray,
    M: int = 32,
    lam: float | None = None,
) -> IntegralReconstruction:
    """
    Run integral method on DDPM dataset-format inputs; return a pixel-grid reconstruction of u.

    The ``fill_mask`` matches ``(gmask | bmask) == 1`` — same region as the dataset's ``u``.
    """
    g0 = inner_curve(M)
    g1 = outer_curve(M)
    blocks = build_kernel_blocks(g0, g1)

    f1, f2 = cauchy_from_boundary_pixels(dirichlet, neumann, bmask, M)
    A, b = assemble_system(g1, blocks, f1, f2, M)
    cond_A = float(np.linalg.cond(A))

    if lam is None:
        result = tikhonov_lcurve(A, b)
        x = result.x
        lam_used = result.lam
    else:
        AtA = A.T @ A
        Atb = A.T @ b
        x = np.linalg.solve(AtA + lam * np.eye(A.shape[1]), Atb)
        lam_used = lam

    psi0 = x[: 2 * M]
    psi1 = x[2 * M :]

    pixel_res = gmask.shape[0]
    x_grid, y_grid = _pixel_grid(pixel_res)
    fill_mask = ((gmask.astype(bool)) | (bmask.astype(bool)))
    u_image = evaluate_on_grid(x_grid, y_grid, fill_mask, g0, g1, psi0, psi1, M)

    return IntegralReconstruction(
        u_image=u_image,
        fill_mask=fill_mask,
        f1=f1,
        f2=f2,
        psi0=psi0,
        psi1=psi1,
        lam=float(lam_used),
        cond_A=cond_A,
    )
