"""Scalar error metrics comparing a surrogate prediction vector against
the held-out ground truth.

`compute_metrics(y_true, y_pred)` returns a :class:`Metrics` record
with absolute (MAE, MSE, RMSE, max_ae), relative (MRE, max_re),
correlation-based (R^2 = Pearson^2) and scale-normalised
(NRMSE = RMSE / std(y_true)) errors plus the indices of the worst-case
points. NRMSE is the primary headline number reported in the
convergence tables.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Metrics:
    """Per-prediction error record produced by :func:`compute_metrics`.

    Fields:
      * `n_test`: number of test points,
      * `mae` / `mse` / `rmse`: mean / mean-squared / root-mean-squared
        absolute error,
      * `mre`: mean relative error (|err| / max(1, |y_true|)),
      * `max_ae` / `i_max_ae`: worst absolute error and its index,
      * `max_re` / `i_max_re`: worst relative error and its index,
      * `r2`: squared Pearson correlation between y_true and y_pred,
      * `nrmse`: RMSE normalised by the std of y_true (scale-free).
    """

    n_test: int
    max_re: float
    i_max_re: int
    max_ae: float
    i_max_ae: int
    mae: float
    mre: float
    mse: float
    rmse: float
    r2: float
    nrmse: float

    def as_dict(self) -> dict:
        """Return metrics as a plain dictionary."""
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Compute absolute, relative and scale-normalized errors."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    absolute_error = np.abs(y_true - y_pred)
    relative_error = absolute_error / np.maximum(1.0, np.abs(y_true))
    i_max_ae = int(np.argmax(absolute_error))
    i_max_re = int(np.argmax(relative_error))
    mse = float(np.mean(absolute_error**2))
    rmse = float(np.sqrt(mse))
    sigma_true = float(np.std(y_true))
    nrmse = rmse / sigma_true if sigma_true > 0.0 else float("nan")

    if sigma_true > 0.0 and np.std(y_pred) > 0.0:
        correlation = float(np.corrcoef(y_true, y_pred)[0, 1])
        r2 = correlation * correlation
    else:
        r2 = float("nan")

    return Metrics(
        n_test=y_true.size,
        max_re=float(relative_error[i_max_re]),
        i_max_re=i_max_re,
        max_ae=float(absolute_error[i_max_ae]),
        i_max_ae=i_max_ae,
        mae=float(np.mean(absolute_error)),
        mre=float(np.mean(relative_error)),
        mse=mse,
        rmse=rmse,
        r2=r2,
        nrmse=nrmse,
    )


def _self_test() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(50.0, 10.0, 200)
    perfect = compute_metrics(y, y.copy())
    noisy = compute_metrics(y, y + rng.normal(0.0, 1.0, 200))
    assert perfect.rmse < 1e-10 and abs(perfect.r2 - 1.0) < 1e-10
    assert noisy.rmse > 0.0 and 0.0 < noisy.r2 < 1.0
    assert noisy.nrmse > 0.0
    print(
        "[metrics] self-test OK "
        f"(RMSE={noisy.rmse:.3f}, NRMSE={noisy.nrmse:.3f}, R2={noisy.r2:.4f})"
    )


if __name__ == "__main__":
    _self_test()
