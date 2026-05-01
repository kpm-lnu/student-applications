"""Boundary curve parametrisations for the doubly-connected domain D = D1 \\ D̄0.

Both Γ0 (inner) and Γ1 (outer) are 2π-periodic smooth closed curves. We expose first and second
derivatives so that the Nyström kernel splittings of §2 can use the closed-form diagonal limits.

Outward unit normal convention: ν is the outward unit normal of the *solution domain D*.
- On Γ1 (outer): points outward of D1, i.e. ν1 = (x1_2', -x1_1') / |x1'|.
- On Γ0 (inner): points outward of D, which means *into* D0, i.e. ν0 = -(x0_2', -x0_1') / |x0'|
  = (-x0_2', x0_1') / |x0'|.

This sign convention matches the paper (eq. 2.5 uses the same outward normal of D on both
boundary components for the double-layer kernel K_{ij}).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Curve:
    """Sampled boundary curve at uniform parameter nodes t_j = j*pi/M, j = 0..2M-1."""

    t: np.ndarray          # (2M,)  parameter nodes
    x: np.ndarray          # (2M, 2) points x(t_j)
    xp: np.ndarray         # (2M, 2) first derivative x'(t_j)
    xpp: np.ndarray        # (2M, 2) second derivative x''(t_j)
    speed: np.ndarray      # (2M,)   |x'(t_j)|
    normal: np.ndarray     # (2M, 2) outward unit normal of D at x(t_j)
    is_outer: bool         # True for Γ1, False for Γ0


def _sample_curve(
    x_fn: Callable[[np.ndarray], np.ndarray],
    xp_fn: Callable[[np.ndarray], np.ndarray],
    xpp_fn: Callable[[np.ndarray], np.ndarray],
    M: int,
    is_outer: bool,
) -> Curve:
    t = np.arange(2 * M) * np.pi / M
    x = x_fn(t)            # (2M, 2)
    xp = xp_fn(t)
    xpp = xpp_fn(t)
    speed = np.linalg.norm(xp, axis=1)
    # Right-hand normal to (x1', x2') is (x2', -x1'); this is outward for a CCW curve.
    rhs_normal = np.stack([xp[:, 1], -xp[:, 0]], axis=1) / speed[:, None]
    normal = rhs_normal if is_outer else -rhs_normal
    return Curve(t=t, x=x, xp=xp, xpp=xpp, speed=speed, normal=normal, is_outer=is_outer)


def outer_curve(M: int = 32) -> Curve:
    """
    Γ1: x1(t) = (1.3 cos t, sin t).

    The curve is traversed counter-clockwise, so the right-hand normal is outward.
    """
    def x_fn(t):
        return np.stack([1.3 * np.cos(t), np.sin(t)], axis=1)

    def xp_fn(t):
        return np.stack([-1.3 * np.sin(t), np.cos(t)], axis=1)

    def xpp_fn(t):
        return np.stack([-1.3 * np.cos(t), -np.sin(t)], axis=1)

    return _sample_curve(x_fn, xp_fn, xpp_fn, M, is_outer=True)


def inner_curve(M: int = 32) -> Curve:
    """
    Γ0: x0(t) = (0.5 cos t, 0.4 sin t − 0.3 sin² t).

    Same parametrisation as in `dataset/generate_harmonic_dataset.py`.
    Traversed counter-clockwise, so the outward normal of *D* (pointing into D0)
    is the negated right-hand normal.
    """
    def x_fn(t):
        return np.stack([0.5 * np.cos(t), 0.4 * np.sin(t) - 0.3 * np.sin(t) ** 2], axis=1)

    def xp_fn(t):
        return np.stack(
            [-0.5 * np.sin(t), 0.4 * np.cos(t) - 0.6 * np.sin(t) * np.cos(t)], axis=1
        )

    def xpp_fn(t):
        # d/dt[0.4 cos t - 0.6 sin t cos t] = -0.4 sin t - 0.6 (cos² t - sin² t)
        return np.stack(
            [-0.5 * np.cos(t), -0.4 * np.sin(t) - 0.6 * (np.cos(t) ** 2 - np.sin(t) ** 2)],
            axis=1,
        )

    return _sample_curve(x_fn, xp_fn, xpp_fn, M, is_outer=False)
