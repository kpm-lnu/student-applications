"""CLI: reconstruct u and ∂u/∂ν on Γ_0 from Cauchy data on Γ_1, and plot results.

Examples:
    python integral_method/run_reconstruction.py --M 32
    python integral_method/run_reconstruction.py --M 32 --noise-pct 3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Allow running as a script from repo root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integral_method.boundary import outer_curve, inner_curve
from integral_method.kernels import build_kernel_blocks
from integral_method.reconstruct import reconstruct_on_inner
from integral_method.synthetic import generate_cauchy_data
from integral_method.system import assemble_system
from integral_method.tikhonov import tikhonov_lcurve


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--M", type=int, default=32, help="half number of quadrature nodes")
    p.add_argument("--noise-pct", type=float, default=0.0, help="Gaussian noise level on Γ_1 data (% of L²-norm)")
    p.add_argument("--lam", type=float, default=None, help="override Tikhonov λ (else L-curve)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="integral_method_outputs")
    p.add_argument("--no-plot", action="store_true", help="skip matplotlib plotting")
    p.add_argument(
        "--eval-grid",
        type=int,
        default=None,
        metavar="PIXEL_RES",
        help="also reconstruct u on a (PIXEL_RES, PIXEL_RES) grid over [-1.5, 1.5]² and save as .npz/.png",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Spurious NumPy 2.x BLAS FPE-flag warnings on matmul; values are finite.
    np.seterr(divide="ignore", over="ignore", invalid="ignore")

    g0 = inner_curve(args.M)
    g1 = outer_curve(args.M)
    blocks = build_kernel_blocks(g0, g1)

    data = generate_cauchy_data(g0, g1, noise_pct=args.noise_pct, seed=args.seed)

    A, b = assemble_system(g1, blocks, data.f1, data.f2, args.M)
    cond = np.linalg.cond(A)

    if args.lam is None:
        result = tikhonov_lcurve(A, b)
        x = result.x
        lam = result.lam
    else:
        # Direct closed-form (A^T A + λ I) x = A^T b
        AtA = A.T @ A
        Atb = A.T @ b
        x = np.linalg.solve(AtA + args.lam * np.eye(A.shape[1]), Atb)
        lam = args.lam
        result = None

    psi0 = x[: 2 * args.M]
    psi1 = x[2 * args.M :]

    u_inner, dudn_inner = reconstruct_on_inner(g0, blocks, psi0, psi1, args.M)

    err_u_inf = float(np.max(np.abs(u_inner - data.u_ref_inner)))
    err_dudn_inf = float(np.max(np.abs(dudn_inner - data.dudn_ref_inner)))
    err_u_l2 = float(np.sqrt(np.mean((u_inner - data.u_ref_inner) ** 2)))
    err_dudn_l2 = float(np.sqrt(np.mean((dudn_inner - data.dudn_ref_inner) ** 2)))

    print(f"M = {args.M}, noise = {args.noise_pct}% , cond(A) = {cond:.3e}")
    print(f"Tikhonov λ = {lam:.3e}")
    print(f"  ‖u  − ũ ‖∞,Γ0    = {err_u_inf:.3e}    L2 = {err_u_l2:.3e}")
    print(f"  ‖∂νu − ∂νũ‖∞,Γ0 = {err_dudn_inf:.3e}    L2 = {err_dudn_l2:.3e}")

    np.savez(
        os.path.join(args.output_dir, "reconstruction.npz"),
        t=g0.t,
        u_inner=u_inner,
        dudn_inner=dudn_inner,
        u_ref=data.u_ref_inner,
        dudn_ref=data.dudn_ref_inner,
        psi0=psi0,
        psi1=psi1,
        lam=lam,
        cond=cond,
        coeffs=data.coeffs,
        noise_pct=args.noise_pct,
    )

    u_image = None
    grid_x = grid_y = grid_mask = None
    if args.eval_grid is not None:
        from grid.compute_grid import compute_geometry_mask, compute_boundary_mask
        from integral_method.reconstruct import evaluate_on_grid

        pixel_res = args.eval_grid
        square_bounds = (-1.5, 1.5, -1.5, 1.5)
        gmask, grid_x, grid_y = compute_geometry_mask(
            outer_curve_x=g1.x[:, 0], outer_curve_y=g1.x[:, 1],
            inner_curves_x=g0.x[:, 0], inner_curves_y=g0.x[:, 1],
            square_bounds=square_bounds, pixel_res=pixel_res, exclude_boundary=True,
        )
        bmask, _, _ = compute_boundary_mask(
            outer_curve_x=g1.x[:, 0], outer_curve_y=g1.x[:, 1],
            inner_curves_x=g0.x[:, 0], inner_curves_y=g0.x[:, 1],
            square_bounds=square_bounds, pixel_res=pixel_res,
        )
        grid_mask = ((gmask | bmask) > 0)
        u_image = evaluate_on_grid(
            grid_x, grid_y, grid_mask, g0, g1, psi0, psi1, args.M,
        )
        np.savez(
            os.path.join(args.output_dir, "reconstruction_grid.npz"),
            u_image=u_image, x_grid=grid_x, y_grid=grid_y,
            gmask=gmask.astype(np.uint8), bmask=bmask.astype(np.uint8),
            pixel_res=pixel_res,
        )
        print(f"  grid:  pixel_res={pixel_res}, filled pixels={int(grid_mask.sum())}")

    if args.no_plot:
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(g0.t, data.u_ref_inner, "k-", label="exact")
    axes[0].plot(g0.t, u_inner, "r--", label="reconstructed")
    axes[0].set_title("u on Γ₀")
    axes[0].set_xlabel("t")
    axes[0].legend()
    axes[1].plot(g0.t, data.dudn_ref_inner, "k-", label="exact")
    axes[1].plot(g0.t, dudn_inner, "r--", label="reconstructed")
    axes[1].set_title("∂u/∂ν on Γ₀")
    axes[1].set_xlabel("t")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "reconstruction.png"), dpi=150)
    plt.close(fig)

    if result is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.loglog(result.residual_norms, result.solution_norms, "b-")
        ax.loglog(
            result.residual_norms[result.corner_index],
            result.solution_norms[result.corner_index],
            "ro",
            label=f"λ* = {lam:.2e}",
        )
        ax.set_xlabel("‖A x − b‖₂")
        ax.set_ylabel("‖x‖₂")
        ax.set_title("L-curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "lcurve.png"), dpi=150)
        plt.close(fig)

    if u_image is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        display = np.where(grid_mask, u_image, np.nan)
        im = ax.imshow(
            display,
            origin="lower",
            extent=(-1.5, 1.5, -1.5, 1.5),
            cmap="viridis",
        )
        ax.plot(g1.x[:, 0], g1.x[:, 1], "k-", lw=1)
        ax.plot(g0.x[:, 0], g0.x[:, 1], "k-", lw=1)
        ax.set_aspect("equal")
        ax.set_title("Reconstructed u on D")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "reconstruction_grid.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
