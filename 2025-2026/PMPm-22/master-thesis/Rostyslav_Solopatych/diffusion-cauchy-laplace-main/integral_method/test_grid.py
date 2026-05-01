"""End-to-end test: reconstruct u on a 2D pixel grid and compare to harmonic ground truth."""
import numpy as np

from grid.compute_grid import compute_geometry_mask
from integral_method.boundary import inner_curve, outer_curve
from integral_method.kernels import build_kernel_blocks
from integral_method.reconstruct import evaluate_on_grid
from integral_method.synthetic import _harmonic_value_and_grad, generate_cauchy_data
from integral_method.system import assemble_system
from integral_method.tikhonov import tikhonov_lcurve


def test_evaluate_on_grid_clean_data():
    M, pixel_res = 32, 64
    g0, g1 = inner_curve(M), outer_curve(M)
    blocks = build_kernel_blocks(g0, g1)
    data = generate_cauchy_data(g0, g1, noise_pct=0.0)
    A, b = assemble_system(g1, blocks, data.f1, data.f2, M)
    res = tikhonov_lcurve(A, b)
    psi0, psi1 = res.x[: 2 * M], res.x[2 * M :]

    gmask, X, Y = compute_geometry_mask(
        outer_curve_x=g1.x[:, 0], outer_curve_y=g1.x[:, 1],
        inner_curves_x=g0.x[:, 0], inner_curves_y=g0.x[:, 1],
        square_bounds=(-1.5, 1.5, -1.5, 1.5),
        pixel_res=pixel_res, exclude_boundary=True,
    )

    u_image = evaluate_on_grid(X, Y, gmask, g0, g1, psi0, psi1, M)

    # Build pixel-aligned ground truth from same harmonic coefficients.
    interior = gmask.astype(bool)
    pts = np.column_stack([X[interior], Y[interior]])
    fake_curve = type(g0)(
        t=np.zeros(pts.shape[0]),
        x=pts,
        xp=np.zeros_like(pts), xpp=np.zeros_like(pts),
        speed=np.ones(pts.shape[0]), normal=np.zeros_like(pts), is_outer=False,
    )
    u_true_vals, _, _ = _harmonic_value_and_grad(fake_curve, data.coeffs)
    u_true_image = np.zeros_like(u_image)
    u_true_image[interior] = u_true_vals

    err = np.abs(u_image - u_true_image)[interior]
    # Trapezoid rule near boundary degrades ~ pixel-band; check 90th percentile is tight.
    assert np.percentile(err, 90) < 1e-3, np.percentile(err, 90)
    # And the bulk median should be machine-precision-ish for clean data.
    assert np.median(err) < 1e-6, np.median(err)
