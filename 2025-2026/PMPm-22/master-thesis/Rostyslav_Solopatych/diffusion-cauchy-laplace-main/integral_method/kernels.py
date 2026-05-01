"""Boundary integral kernels of eq. (2.4)–(2.5) and their Nyström splittings.

We assemble four (2M, 2M) blocks needed for the linear system (eq. 2.11):

* `H01[i,j] = H_{01}(t_i, t_j) = ln 1/|x_1(t_i) − x_0(t_j)|`            -- smooth, off-diagonal
* `H11_smooth[i,j] = H̃_{11}(t_i, t_j)`  (smooth remainder of self-kernel; diagonal handled below)
* `K01[i,j] = (x_0(t_j) − x_1(t_i)) · ν(x_1(t_i)) / |x_1(t_i) − x_0(t_j)|²`   -- smooth
* `K11[i,j] = (x_1(t_j) − x_1(t_i)) · ν(x_1(t_i)) / |x_1(t_i) − x_1(t_j)|²`   -- smooth across diagonal
        with diagonal limit `K11[i,i] = x_1''(t_i) · ν(x_1(t_i)) / (2 |x_1'(t_i)|²)`.

The same builder also returns the corresponding blocks needed to evaluate the trace on Γ_0
(the §2 Remark): `H00_smooth, H10, K00, K10`.

All kernels follow the paper convention: the first index of `H_{ij}, K_{ij}` is the *source*
curve (the one being integrated over) and the second is the *target* curve where the kernel is
evaluated; ν is the outward-of-D normal at the target point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import Curve


@dataclass
class KernelBlocks:
    # source-Γ0, target-Γ1
    H01: np.ndarray
    K01: np.ndarray
    # source-Γ1, target-Γ1  (self-kernels: H11 split, K11 has diagonal limit)
    H11_smooth: np.ndarray
    K11: np.ndarray
    # source-Γ0, target-Γ0  (self-kernels for Γ0 trace)
    H00_smooth: np.ndarray
    K00: np.ndarray
    # source-Γ1, target-Γ0
    H10: np.ndarray
    K10: np.ndarray


def _smooth_log_kernel(target: Curve, source: Curve) -> np.ndarray:
    """H[i,j] = ln(1 / |x_target(t_i) − x_source(t_j)|).  Source curve != target curve."""
    diff = target.x[:, None, :] - source.x[None, :, :]      # (2M, 2M, 2)
    r2 = np.sum(diff * diff, axis=-1)
    return -0.5 * np.log(r2)


def _double_layer_offsource(target: Curve, source: Curve) -> np.ndarray:
    """K[i,j] = (x_source(t_j) − x_target(t_i)) · ν(x_target(t_i)) / |·|².  Different curves."""
    diff = source.x[None, :, :] - target.x[:, None, :]      # (2M, 2M, 2)
    r2 = np.sum(diff * diff, axis=-1)
    num = diff[..., 0] * target.normal[:, None, 0] + diff[..., 1] * target.normal[:, None, 1]
    return num / r2


def _self_log_smooth(curve: Curve) -> np.ndarray:
    """H̃_{ii}(t,τ) = 0.5 ln((4/e) sin²((t-τ)/2) / |x(t)-x(τ)|²) for t≠τ;
    H̃_{ii}(t,t) = 0.5 ln(1/(e |x'(t)|²)).
    """
    t = curve.t
    n = t.size
    diff_t = t[:, None] - t[None, :]
    sin_sq = np.sin(diff_t / 2.0) ** 2
    diff_x = curve.x[:, None, :] - curve.x[None, :, :]
    r2 = np.sum(diff_x * diff_x, axis=-1)
    out = np.empty_like(r2)
    off = ~np.eye(n, dtype=bool)
    # off-diagonal: 0.5 ln((4/e) sin² / |Δx|²) = 0.5 (ln(4/e) + ln sin² − ln r²)
    out[off] = 0.5 * (np.log(4.0 / np.e) + np.log(sin_sq[off]) - np.log(r2[off]))
    # diagonal: 0.5 ln(1 / (e |x'|²))
    diag = 0.5 * np.log(1.0 / (np.e * curve.speed ** 2))
    np.fill_diagonal(out, diag)
    return out


def _self_double_layer(curve: Curve) -> np.ndarray:
    """K_{ii}(t,τ) = (x(τ) − x(t)) · ν(t) / |x(t) − x(τ)|²;
    diagonal limit K_{ii}(t,t) = x''(t) · ν(t) / (2 |x'(t)|²).
    """
    n = curve.t.size
    diff = curve.x[None, :, :] - curve.x[:, None, :]   # x(τ_j) - x(t_i)
    r2 = np.sum(diff * diff, axis=-1)
    num = diff[..., 0] * curve.normal[:, None, 0] + diff[..., 1] * curve.normal[:, None, 1]
    out = np.empty_like(num)
    off = ~np.eye(n, dtype=bool)
    out[off] = num[off] / r2[off]
    diag = (curve.xpp[:, 0] * curve.normal[:, 0] + curve.xpp[:, 1] * curve.normal[:, 1]) / (
        2.0 * curve.speed ** 2
    )
    np.fill_diagonal(out, diag)
    return out


def build_kernel_blocks(gamma0: Curve, gamma1: Curve) -> KernelBlocks:
    """Assemble all kernel blocks needed for system assembly and Γ0 trace reconstruction."""
    return KernelBlocks(
        H01=_smooth_log_kernel(target=gamma1, source=gamma0),
        K01=_double_layer_offsource(target=gamma1, source=gamma0),
        H11_smooth=_self_log_smooth(gamma1),
        K11=_self_double_layer(gamma1),
        H00_smooth=_self_log_smooth(gamma0),
        K00=_self_double_layer(gamma0),
        H10=_smooth_log_kernel(target=gamma0, source=gamma1),
        K10=_double_layer_offsource(target=gamma0, source=gamma1),
    )
