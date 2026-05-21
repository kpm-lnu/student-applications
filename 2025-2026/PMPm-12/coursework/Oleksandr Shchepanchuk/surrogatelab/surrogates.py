"""Scalar-output surrogate models used in the adaptive-sampling study.

Two interpolants share a `fit / predict / standardize_target / latent`
interface:
  * :class:`RBFSurrogate` - radial-basis-function interpolant. Registered
    kernels in :data:`KERNELS`: gaussian, inverse_multiquadric, multiquadric
    (shape-parameter families) and cubic, thin_plate, linear (scale-free).
    Shape parameter eps is chosen by Rippa-Fasshauer LOOCV over
    :data:`DEFAULT_EPS_GRID` when `eps=None`. Optional log-target
    fitting is appropriate for QoIs spanning several orders of magnitude.
  * :class:`KrigingSurrogate` - sklearn GP with ARD-Matern kernel,
    median-pairwise-distance init for length-scales, multi-start MLE.

Both expose `power_function`-style uncertainty / residual quantities
consumed by the greedy adaptive samplers in :mod:`.sampling`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Kernel:
    """Radial kernel record bundling the evaluator with its metadata.

    Fields:
      * `name`: registry key (e.g. ``"gaussian"``, ``"cubic"``),
      * `func(d, eps) -> phi(d)`: vectorised kernel evaluator on distances,
      * `phi0`: phi at distance 0 (used by the power-function formula),
      * `uses_shape`: whether eps controls the kernel (False for cubic /
        thin-plate / linear, which are scale-free and skip the
        LOOCV-over-eps search).
    """

    name: str
    func: Callable[[np.ndarray, float], np.ndarray]
    phi0: float
    uses_shape: bool

    def __call__(self, distances: np.ndarray, eps: float) -> np.ndarray:
        """Evaluate phi(||x - y||)."""
        return self.func(np.asarray(distances, dtype=float), eps)


def _gaussian(distances: np.ndarray, eps: float) -> np.ndarray:
    return np.exp(-(eps * distances) ** 2)


def _inverse_multiquadric(distances: np.ndarray, eps: float) -> np.ndarray:
    return 1.0 / np.sqrt(1.0 + (eps * distances) ** 2)


def _multiquadric(distances: np.ndarray, eps: float) -> np.ndarray:
    return np.sqrt(1.0 + (eps * distances) ** 2)


def _cubic(distances: np.ndarray, eps: float) -> np.ndarray:
    return distances**3


def _thin_plate(distances: np.ndarray, eps: float) -> np.ndarray:
    values = np.zeros_like(distances)
    positive = distances > 0.0
    values[positive] = distances[positive] ** 2 * np.log(distances[positive])
    return values


def _linear(distances: np.ndarray, eps: float) -> np.ndarray:
    return distances


KERNELS: dict[str, Kernel] = {
    "gaussian": Kernel("gaussian", _gaussian, 1.0, True),
    "inverse_multiquadric": Kernel(
        "inverse_multiquadric",
        _inverse_multiquadric,
        1.0,
        True,
    ),
    "multiquadric": Kernel("multiquadric", _multiquadric, 1.0, True),
    "cubic": Kernel("cubic", _cubic, 0.0, False),
    "thin_plate": Kernel("thin_plate", _thin_plate, 0.0, False),
    "linear": Kernel("linear", _linear, 0.0, False),
}

DEFAULT_EPS_GRID = np.logspace(-1.0, 1.3, 28)


def get_kernel(name: str) -> Kernel:
    """Return a registered radial kernel."""
    if name not in KERNELS:
        raise KeyError(f"unknown kernel '{name}'. Available: {sorted(KERNELS)}")
    return KERNELS[name]


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between rows of two arrays."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    squared = (
        np.sum(a * a, axis=1, keepdims=True)
        + np.sum(b * b, axis=1)
        - 2.0 * a @ b.T
    )
    return np.sqrt(np.maximum(squared, 0.0))


@dataclass
class RBFSurrogate:
    """RBF interpolant on unit-cube inputs with Tikhonov nugget,
    target standardisation and optional log-transform of the QoI.

    Pipeline applied on :meth:`fit`:
      1. Optional ``log_transform``: targets must be positive; the
         interpolation is performed on ``log(y)`` (appropriate for QoIs
         spanning orders of magnitude).
      2. Target standardisation: subtract mean, divide by std so the
         system matrix is well-scaled.
      3. Kernel selection via :func:`get_kernel`; for shape-parameter
         kernels the shape `eps` is either taken from the user or chosen
         by Rippa-Fasshauer LOOCV over `eps_grid`.
      4. Solve ``(Phi + nugget * I) alpha = z`` via Cholesky/lstsq.

    Exposes :meth:`predict`, :meth:`latent`, :meth:`power_function`,
    :meth:`loo_residuals`, :meth:`condition_number` - the latter three
    feed the greedy adaptive samplers.
    """

    kernel: str = "gaussian"
    eps: float | None = None
    eps_grid: np.ndarray = field(default_factory=lambda: DEFAULT_EPS_GRID)
    nugget: float = 1e-10
    log_transform: bool = True
    standardize: bool = True
    _fitted: bool = field(default=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFSurrogate":
        """Fit the interpolant on unit-cube inputs and scalar targets."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have matching first dimension")

        kernel = get_kernel(self.kernel)
        self._kernel_obj = kernel
        self.X_ = X
        self.y_ = y
        transformed = self._transform_target(y)
        self._mu = float(np.mean(transformed)) if self.standardize else 0.0
        self._sigma = float(np.std(transformed)) if self.standardize else 1.0
        if self._sigma < 1e-14:
            self._sigma = 1.0
        standardized = (transformed - self._mu) / self._sigma
        self._z_std = standardized

        if kernel.uses_shape:
            self.eps_used_ = (
                float(self.eps)
                if self.eps is not None
                else self._select_eps(X, standardized, kernel)
            )
        else:
            self.eps_used_ = 1.0

        self._solve(X, standardized, kernel, self.eps_used_)
        self._fitted = True
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Evaluate the surrogate at new unit-cube points."""
        self._check_fitted()
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        kernel_matrix = self._kernel_obj(
            pairwise_distances(X_new, self.X_),
            self.eps_used_,
        )
        standardized = kernel_matrix @ self._coeffs
        transformed = self._mu + self._sigma * standardized
        return np.exp(transformed) if self.log_transform else transformed

    def latent(self, X_new: np.ndarray) -> np.ndarray:
        """Prediction in the standardized transformed target space."""
        self._check_fitted()
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        kernel_matrix = self._kernel_obj(
            pairwise_distances(X_new, self.X_),
            self.eps_used_,
        )
        return kernel_matrix @ self._coeffs

    def standardize_target(self, y: np.ndarray) -> np.ndarray:
        """Map raw targets to the fitted standardized target space."""
        self._check_fitted()
        transformed = self._transform_target(np.asarray(y, dtype=float).ravel())
        return (transformed - self._mu) / self._sigma

    def power_function(self, X_new: np.ndarray) -> np.ndarray:
        """Kernel power function at new unit-cube points."""
        self._check_fitted()
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        kernel_matrix = self._kernel_obj(
            pairwise_distances(X_new, self.X_),
            self.eps_used_,
        )
        solved = _solve_linear(self._A, kernel_matrix.T)
        quadratic = np.einsum("mn,nm->m", kernel_matrix, solved)
        return np.sqrt(np.maximum(self._kernel_obj.phi0 - quadratic, 0.0))

    def loo_residuals(self) -> np.ndarray:
        """Rippa leave-one-out residuals at the training points."""
        self._check_fitted()
        inverse_diag = _inverse_diagonal(self._A)
        inverse_diag = np.where(np.abs(inverse_diag) > 1e-300, inverse_diag, 1.0)
        return self._coeffs / inverse_diag

    @property
    def condition_number(self) -> float:
        """Condition number of the unregularized kernel matrix."""
        self._check_fitted()
        return self._cond

    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        if self.log_transform:
            if np.any(y <= 0.0):
                raise ValueError("log_transform=True requires positive y")
            return np.log(y)
        return y.copy()

    def _select_eps(self, X: np.ndarray, z_std: np.ndarray, kernel: Kernel) -> float:
        best_eps = float(self.eps_grid[0])
        best_loo = np.inf
        for eps in self.eps_grid:
            loo = self._loo_rmse_for_eps(X, z_std, kernel, float(eps))
            if loo < best_loo:
                best_eps = float(eps)
                best_loo = loo
        self.loo_rmse_ = best_loo
        return best_eps

    def _loo_rmse_for_eps(
        self,
        X: np.ndarray,
        z_std: np.ndarray,
        kernel: Kernel,
        eps: float,
    ) -> float:
        n_points = X.shape[0]
        kernel_matrix = kernel(pairwise_distances(X, X), eps)
        system_matrix = kernel_matrix + self.nugget * np.eye(n_points)
        try:
            coefficients = _solve_linear(system_matrix, z_std)
            inverse_diag = _inverse_diagonal(system_matrix)
        except np.linalg.LinAlgError:
            return np.inf
        if np.any(np.abs(inverse_diag) <= 1e-300) or not np.all(np.isfinite(inverse_diag)):
            return np.inf
        residuals = coefficients / inverse_diag
        return float(np.sqrt(np.mean(residuals * residuals)))

    def _solve(
        self,
        X: np.ndarray,
        z_std: np.ndarray,
        kernel: Kernel,
        eps: float,
    ) -> None:
        n_points = X.shape[0]
        kernel_matrix = kernel(pairwise_distances(X, X), eps)
        system_matrix = kernel_matrix + self.nugget * np.eye(n_points)
        self._coeffs = _solve_linear(system_matrix, z_std)
        self._A = system_matrix
        self._cond = float(np.linalg.cond(kernel_matrix))

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("surrogate is not fitted")


@dataclass
class KrigingSurrogate:
    """Gaussian process surrogate using an ARD Matern kernel.

    Provides the same public interface as :class:`RBFSurrogate` (``fit``,
    ``predict``, ``standardize_target``, ``latent``) for the cross-method
    comparison in section 5.6 of the coursework. Wraps sklearn's
    ``GaussianProcessRegressor`` with an anisotropic Matern (nu=5/2)
    kernel and MLE optimisation of the hyperparameters
    (Rasmussen-Williams 2006, MIT Press).
    """

    length_scale_bounds: tuple[float, float] = (1e-3, 1e3)
    nu: float = 2.5
    nugget: float = 1e-10
    log_transform: bool = True
    n_restarts: int = 15

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KrigingSurrogate":
        """Fit the GP on unit-cube inputs and scalar targets."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel as ConstantKern,
            Matern,
        )

        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have matching first dimension")
        if self.log_transform and np.any(y <= 0.0):
            raise ValueError("log_transform=True requires positive y")

        transformed = np.log(y) if self.log_transform else y.copy()
        self._mu = float(np.mean(transformed))
        sigma = float(np.std(transformed))
        self._sigma = sigma if sigma > 1e-14 else 1.0
        z_std = (transformed - self._mu) / self._sigma

        d = X.shape[1]
        # Rasmussen-Williams §5.4.1: per-dimension median pairwise distance is a
        # robust initial guess for the ARD length-scales when MLE is multi-modal.
        if X.shape[0] >= 2:
            triu = np.triu_indices(X.shape[0], k=1)
            median_dists = np.empty(d)
            for i in range(d):
                col = X[:, i:i + 1]
                pairwise = np.abs(col - col.T)[triu]
                median_dists[i] = np.median(pairwise)
            median_dists = np.where(median_dists > 1e-6, median_dists, 0.1)
        else:
            median_dists = np.full(d, 0.1)
        kernel = ConstantKern(1.0, (1e-3, 1e3)) * Matern(
            length_scale=median_dists,
            length_scale_bounds=self.length_scale_bounds,
            nu=self.nu,
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.nugget,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=False,
        ).fit(X, z_std)
        self.X_ = X
        self.y_ = y
        self._fitted = True
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict the QoI at new unit-cube points (original scale)."""
        self._check_fitted()
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        z_std_pred = self._gp.predict(X_new)
        z_pred = z_std_pred * self._sigma + self._mu
        return np.exp(z_pred) if self.log_transform else z_pred

    def latent(self, X_new: np.ndarray) -> np.ndarray:
        """Standardised log/raw target prediction matching RBF semantics."""
        self._check_fitted()
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        return self._gp.predict(X_new)

    def standardize_target(self, y: np.ndarray) -> np.ndarray:
        """Map raw targets to the fitted standardised target space."""
        self._check_fitted()
        transformed = np.log(y) if self.log_transform else np.asarray(y, dtype=float)
        return (transformed - self._mu) / self._sigma

    def _check_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("surrogate is not fitted")


def _solve_linear(system_matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(system_matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system_matrix, rhs, rcond=None)[0]


def _inverse_diagonal(system_matrix: np.ndarray) -> np.ndarray:
    inverse = _solve_linear(system_matrix, np.eye(system_matrix.shape[0]))
    return np.diag(inverse)


def _self_test() -> None:
    rng = np.random.default_rng(0)

    def target(X: np.ndarray) -> np.ndarray:
        return 2.0 + np.sin(2.0 * np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1])

    X = rng.uniform(0.0, 1.0, (45, 2))
    y = target(X)
    X_test = rng.uniform(0.0, 1.0, (300, 2))
    y_test = target(X_test)

    for kernel_name in ("gaussian", "cubic", "thin_plate", "inverse_multiquadric"):
        surrogate = RBFSurrogate(kernel=kernel_name, log_transform=True).fit(X, y)
        error = np.sqrt(np.mean((surrogate.predict(X_test) - y_test) ** 2)) / np.std(y_test)
        print(
            f"[surrogate] {kernel_name:22s} "
            f"NRMSE={error:.2e} cond(Phi)={surrogate.condition_number:.1e}"
        )
        assert error < 0.2

    gaussian = RBFSurrogate(kernel="gaussian").fit(X, y)
    power_train = gaussian.power_function(X)
    power_test = gaussian.power_function(X_test)
    assert power_train.max() < 1e-2 and power_train.max() < power_test.mean()
    assert gaussian.loo_residuals().shape == (45,)
    print("[surrogate] self-test OK")


if __name__ == "__main__":
    _self_test()
