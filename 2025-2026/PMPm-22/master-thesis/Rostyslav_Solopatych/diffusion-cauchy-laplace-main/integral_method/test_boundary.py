"""Tests for boundary curve sampling and outward-of-D normal sign convention."""
import numpy as np

from integral_method.boundary import inner_curve, outer_curve


def test_outer_normal_points_outward():
    g1 = outer_curve(M=32)
    # On an ellipse centred at origin, outward normal · position > 0.
    dot = np.sum(g1.normal * g1.x, axis=1)
    assert np.all(dot > 0)
    # Unit length.
    np.testing.assert_allclose(np.linalg.norm(g1.normal, axis=1), 1.0, atol=1e-12)


def test_inner_normal_points_into_D0():
    g0 = inner_curve(M=32)
    # The Γ_0 curve roughly surrounds the origin (since 0.4 sin(0) - 0.3 sin²(0) = 0 etc.).
    # Outward-of-D normal must point *into* D_0 (toward the curve's interior centroid).
    centroid = g0.x.mean(axis=0)
    inward_pointer = centroid - g0.x       # from boundary toward centroid
    dot = np.sum(g0.normal * inward_pointer, axis=1)
    assert np.all(dot > 0)


def test_derivatives_finite_difference():
    for curve_fn in (inner_curve, outer_curve):
        g = curve_fn(M=64)
        # Compare x' against centred difference of x along the parameter.
        dt = g.t[1] - g.t[0]
        xp_fd = (np.roll(g.x, -1, axis=0) - np.roll(g.x, 1, axis=0)) / (2 * dt)
        np.testing.assert_allclose(xp_fd, g.xp, atol=5e-3)
