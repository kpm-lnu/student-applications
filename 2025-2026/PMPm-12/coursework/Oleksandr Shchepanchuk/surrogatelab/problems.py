"""Parametric forward PDE / ODE / analytic models exposed via a common
:class:`Problem` interface.

Each subclass defines:
  * `bounds`: physical parameter box (shape ``(d, 2)``),
  * `qoi_names`: ordered scalar outputs,
  * `solve(mu)` and `evaluate(mu)`: ground-truth PDE / ODE solution and
    the cached scalar QoIs at one physical parameter point,
  * `as_unit_cube_batch(qoi)`: vectorised QoI-on-[0,1]^d callable used by
    the samplers and surrogate fits.

Provided models:
  * :class:`ReactionDiffusionProblem` (1D + time, 2 species, 2D/4D),
  * :class:`HeatAMICon_FD`, :class:`AdvDiffAMICon_FD` (1D steady FD,
    AMICon-2026 reference),
  * :class:`HeatAMIConProblem`, :class:`AdvDiffAMIConProblem` (deprecated
    P1-FEM versions of the AMICon problems, kept for diagnostics),
  * :class:`BraninProblem` (2D analytic test function),
  * :class:`HeatProblem` (deprecated placeholder).
"""
from __future__ import annotations

import math
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp, trapezoid

from . import fem


class Problem(ABC):
    """Common interface for every parametric forward model.

    Subclasses must expose `bounds` (physical parameter box, shape
    ``(d, 2)``), `qoi_names` (ordered scalar outputs), and `evaluate(mu)`
    (returning a ``{qoi_name: value}`` dict for one physical parameter
    point). The base class adds:
      * `dim`, `to_unit_cube`, `from_unit_cube`: domain helpers,
      * `evaluate_qoi`: batched evaluation in unit-cube coordinates,
      * `as_unit_cube_batch(qoi)`: cached vectorised callable used by the
        samplers and surrogate fits.
    """

    name: str = "abstract"
    param_names: list[str] = []
    bounds: np.ndarray = np.zeros((0, 2))
    qoi_names: list[str] = []

    @property
    def dim(self) -> int:
        """Number of active parameters."""
        return len(self.param_names)

    def to_unit(self, mu: np.ndarray) -> np.ndarray:
        """Map physical parameters to [0, 1]^d."""
        mu = np.atleast_2d(np.asarray(mu, dtype=float))
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return (mu - lower) / (upper - lower)

    def from_unit(self, x: np.ndarray) -> np.ndarray:
        """Map [0, 1]^d points to physical parameters."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return lower + x * (upper - lower)

    @abstractmethod
    def evaluate(self, mu: np.ndarray) -> dict[str, float]:
        """Evaluate all QoIs at one physical parameter point."""

    def evaluate_unit(self, x: np.ndarray, qoi: str) -> np.ndarray:
        """Evaluate one QoI for unit-cube parameter points."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        parameters = self.from_unit(x)
        return np.array([self.evaluate(parameters[row])[qoi] for row in range(len(parameters))])

    def forward(self, qoi: str) -> Callable[[np.ndarray], np.ndarray]:
        """Return a cached batch evaluator for one QoI on the unit cube."""
        cache: dict[tuple[float, ...], dict[str, float]] = {}

        def evaluate_batch(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(np.asarray(x, dtype=float))
            parameters = self.from_unit(x)
            values = np.empty(parameters.shape[0])
            for row, parameter in enumerate(parameters):
                key = tuple(np.round(parameter, 10))
                if key not in cache:
                    cache[key] = self.evaluate(parameter)
                values[row] = cache[key][qoi]
            return values

        return evaluate_batch


class ReactionDiffusionProblem(Problem):
    """Two-species reaction-diffusion PDE on x in [-2, 2], t in [0, 25].

    Method of lines (P1 FEM in space, ``solve_ivp`` LSODA in time):
        dy1/dt = D1 * d2y1/dx2 + p1*y1 + p2*y1^2 + p3*y1*y2
        dy2/dt = D2 * d2y2/dx2 + p4*y2 + p5*y2^2 + p6*y1*y2
    Initial condition: y1(x, 0) = Y01 * exp(-a_init * x^2), y2(x, 0) = Y02.
    Boundary condition: zero-flux Neumann at x = +-2.

    Six physical reaction parameters p1..p6 live in per-component bands
    [p_minus, p_plus] (class attributes). Only an `active` subset is
    parameterised: default 2D uses (p1, p3); the 4D variant created via
    :meth:`with_4d_params` uses (p1, p3, p4, p6).

    Three scalar QoIs are exposed:
        J  = double integral of y1^2 + y2^2 over the whole space-time slab,
        J1 = integral over x of y1(x, te)^2  (final-time spatial integral),
        J2 = integral over t of y1(0, t)^2  (midpoint time integral).

    Diffusivities are fixed (D1 = 0, D2 = 0.2), so y1 is reaction-only
    and y2 carries the spatial coupling - this is what makes the QoIs
    non-trivially parameter-dependent.
    """

    name = "reaction_diffusion_v9"
    qoi_names = ["J", "J2"]

    t0, te = 0.0, 25.0
    x0, xe = -2.0, 2.0
    Y01, Y02 = 2.0, 4.0
    a_init = 5.0
    D1, D2 = 0.0, 0.2
    p_base = np.array([-1.0, 0.0, 0.2, 3.0, 0.0, -0.5])
    p_minus = np.array([-1.4, 0.0, 0.15, 2.0, 0.0, -0.7])
    p_plus = np.array([-0.6, 0.0, 0.25, 4.0, 0.0, -0.3])

    def __init__(
        self,
        pde_nx: int = 21,
        active: tuple[int, ...] | None = None,
        param_indices: tuple[int, ...] = (0, 2),
        mass: str = "consistent",
    ) -> None:
        if mass not in {"consistent", "lumped"}:
            raise ValueError("mass must be 'consistent' or 'lumped'")
        self.pde_nx = int(pde_nx)
        if self.pde_nx < 2:
            raise ValueError("pde_nx must be at least 2")
        self.param_indices = tuple(active if active is not None else param_indices)
        self.active = self.param_indices
        all_names = ["p1", "p2", "p3", "p4", "p5", "p6"]
        if any(index < 0 or index >= len(all_names) for index in self.param_indices):
            raise ValueError("parameter index out of range")

        self.mass = mass
        self.param_names = [all_names[index] for index in self.param_indices]
        self.bounds = np.array(
            [[self.p_minus[index], self.p_plus[index]] for index in self.param_indices],
            dtype=float,
        )
        self._eval_cache: dict[tuple[float, ...], dict[str, float]] = {}
        self.n_solver_calls = 0
        self._fast_x = np.linspace(self.x0, self.xe, self.pde_nx)
        self._fast_dx = self._fast_x[1] - self._fast_x[0]
        self._fast_laplacian = fem.laplacian_neumann(
            self.pde_nx,
            self._fast_dx,
            mass=self.mass,
        )
        self._fast_midpoint_index = int(np.argmin(np.abs(self._fast_x)))

    @classmethod
    def with_4d_params(
        cls,
        pde_nx: int = 21,
        mass: str = "consistent",
    ) -> "ReactionDiffusionProblem":
        """Create the p1, p3, p4, p6 parameterization."""
        return cls(pde_nx=pde_nx, param_indices=(0, 2, 3, 5), mass=mass)

    def parameter_bounds(self) -> np.ndarray:
        """Bounds for the selected active parameters."""
        return self.bounds.copy()

    @property
    def L(self) -> float:
        """Spatial-domain length."""
        return self.xe - self.x0

    def full_params(self, mu_active: np.ndarray) -> np.ndarray:
        """Insert active parameters into the full six-parameter vector."""
        parameters = self.p_base.copy()
        active_values = np.asarray(mu_active, dtype=float).ravel()
        if active_values.shape != (len(self.param_indices),):
            raise ValueError("mu has wrong dimension")
        for local_index, parameter_index in enumerate(self.param_indices):
            parameters[parameter_index] = active_values[local_index]
        return parameters

    @staticmethod
    def reaction(
        y1: np.ndarray | float,
        y2: np.ndarray | float,
        parameters: np.ndarray,
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """Reaction terms of the resource-consumer system."""
        p1, p2, p3, p4, p5, p6 = parameters
        f1 = p1 * y1 + p2 * y1 * y1 + p3 * y1 * y2
        f2 = p4 * y2 + p5 * y2 * y2 + p6 * y1 * y2
        return f1, f2

    def solve_ode(
        self,
        parameters: np.ndarray | None = None,
        n_t: int = 1201,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the spatially homogeneous ODE system."""
        parameters = self.p_base if parameters is None else parameters
        time_points = np.linspace(self.t0, self.te, n_t)
        solution = solve_ivp(
            lambda _time, state: self.reaction(state[0], state[1], parameters),
            (self.t0, self.te),
            [self.Y01, self.Y02],
            t_eval=time_points,
            method="DOP853",
            rtol=1e-9,
            atol=1e-11,
        )
        if not solution.success:
            raise RuntimeError(f"ODE solve failed: {solution.message}")
        return solution.t, solution.y

    def ode_functional_J(self, parameters: np.ndarray | None = None) -> float:
        """Time integral of y1^2 + y2^2 for the ODE model."""
        parameters = self.p_base if parameters is None else parameters

        def rhs(_time: float, state: np.ndarray) -> list[float]:
            f1, f2 = self.reaction(state[0], state[1], parameters)
            return [f1, f2, state[0] * state[0] + state[1] * state[1]]

        solution = solve_ivp(
            rhs,
            (self.t0, self.te),
            [self.Y01, self.Y02, 0.0],
            method="DOP853",
            rtol=1e-9,
            atol=1e-11,
        )
        if not solution.success:
            raise RuntimeError(f"ODE-J solve failed: {solution.message}")
        return float(solution.y[2, -1])

    def jacobian(
        self,
        y1: float,
        y2: float,
        parameters: np.ndarray | None = None,
    ) -> np.ndarray:
        """Jacobian of the reaction vector field at one state."""
        parameters = self.p_base if parameters is None else parameters
        p1, p2, p3, p4, p5, p6 = parameters
        return np.array(
            [
                [p1 + 2.0 * p2 * y1 + p3 * y2, p3 * y1],
                [p6 * y2, p4 + 2.0 * p5 * y2 + p6 * y1],
            ],
            dtype=float,
        )

    def equilibria(self, parameters: np.ndarray | None = None) -> list[tuple[float, float]]:
        """Origin and coexistence equilibria of the reaction system."""
        parameters = self.p_base if parameters is None else parameters
        p1, _p2, p3, p4, _p5, p6 = parameters
        equilibria = [(0.0, 0.0)]
        if p3 != 0.0 and p6 != 0.0:
            equilibria.append((-p4 / p6, -p1 / p3))
        return equilibria

    def equilibrium_analysis(self, parameters: np.ndarray | None = None) -> list[dict]:
        """Classify equilibria by reaction-Jacobian eigenvalues."""
        parameters = self.p_base if parameters is None else parameters
        rows = []
        for y1, y2 in self.equilibria(parameters):
            jacobian = self.jacobian(y1, y2, parameters)
            eigenvalues = np.linalg.eigvals(jacobian)
            rows.append(
                {
                    "y1": y1,
                    "y2": y2,
                    "jacobian": jacobian,
                    "lambda1": eigenvalues[0],
                    "lambda2": eigenvalues[1],
                    "type": _classify_eigenvalues(eigenvalues),
                }
            )
        return rows

    def stationary_analysis(self, parameters: np.ndarray | None = None) -> dict:
        """Spatial first-order analysis of t-independent solutions."""
        parameters = self.p_base if parameters is None else parameters
        omega_squared = parameters[3] / self.D2
        omega = math.sqrt(abs(omega_squared))
        return {
            "constant_solutions": self.equilibria(parameters),
            "spatial_omega": omega,
            "spatial_eigenvalues": (complex(0.0, omega), complex(0.0, -omega)),
            "spatial_point_type": "centre" if omega_squared > 0.0 else "saddle",
            "neumann_mode_frequencies": [
                mode * math.pi / self.L for mode in range(1, 7)
            ],
        }

    def solve_pde(
        self,
        parameters: np.ndarray | None = None,
        n_x: int = 161,
        n_t: int = 301,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the full method-of-lines PDE system."""
        parameters = self.p_base if parameters is None else parameters
        x = np.linspace(self.x0, self.xe, n_x)
        dx = x[1] - x[0]
        laplacian = fem.laplacian_neumann(n_x, dx, mass=self.mass)
        time_points = np.linspace(self.t0, self.te, n_t)
        initial_profile = np.exp(-self.a_init * x * x)
        initial_state = np.r_[self.Y01 * initial_profile, self.Y02 * initial_profile]

        def rhs(_time: float, state: np.ndarray) -> np.ndarray:
            y1 = state[:n_x]
            y2 = state[n_x:]
            f1, f2 = self.reaction(y1, y2, parameters)
            dy1 = self.D1 * (laplacian @ y1) + f1
            dy2 = self.D2 * (laplacian @ y2) + f2
            return np.r_[dy1, dy2]

        solution = solve_ivp(
            rhs,
            (self.t0, self.te),
            initial_state,
            t_eval=time_points,
            method="LSODA",
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise RuntimeError(f"PDE solve failed: {solution.message}")
        return x, solution.t, solution.y.reshape(2, n_x, -1)

    def pde_functionals_from_solution(
        self,
        x: np.ndarray,
        t: np.ndarray,
        solution: np.ndarray,
    ) -> tuple[float, float]:
        """Final-time space integral J1 and midpoint time integral J2."""
        y1 = solution[0]
        y2 = solution[1]
        final_energy = y1[:, -1] ** 2 + y2[:, -1] ** 2
        midpoint_y1 = np.array([np.interp(0.0, x, y1[:, step]) for step in range(len(t))])
        midpoint_y2 = np.array([np.interp(0.0, x, y2[:, step]) for step in range(len(t))])
        J1 = float(trapezoid(final_energy, x))
        J2 = float(trapezoid(midpoint_y1 * midpoint_y1 + midpoint_y2 * midpoint_y2, t))
        return J1, J2

    def evaluate(self, mu: np.ndarray) -> dict[str, float]:
        """Evaluate J, J1 and J2 at one active-parameter point."""
        key = tuple(np.round(np.asarray(mu, dtype=float).ravel(), 10))
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached

        parameters = self.full_params(np.asarray(mu, dtype=float))
        J = self.ode_functional_J(parameters)
        _J1, J2 = self._pde_functionals_fast(parameters)
        result = {"J": J, "J2": J2}
        self._eval_cache[key] = result
        self.n_solver_calls += 1
        return result

    def save_cache(self, path: str | Path) -> None:
        """Persist compatible forward-model evaluations."""
        payload = {"meta": self._cache_metadata(), "values": self._eval_cache}
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)

    def load_cache(self, path: str | Path) -> int:
        """Load compatible cached forward-model evaluations."""
        path = Path(path)
        if not path.exists():
            return 0
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or payload.get("meta") != self._cache_metadata():
            self._eval_cache = {}
            return 0
        self._eval_cache = payload.get("values", {})
        return len(self._eval_cache)

    def _pde_functionals_fast(self, parameters: np.ndarray) -> tuple[float, float]:
        n_x = self.pde_nx
        x = self._fast_x
        laplacian = self._fast_laplacian
        midpoint_index = self._fast_midpoint_index
        initial_profile = np.exp(-self.a_init * x * x)
        initial_state = np.r_[self.Y01 * initial_profile, self.Y02 * initial_profile, 0.0]

        def rhs(_time: float, state: np.ndarray) -> np.ndarray:
            y1 = state[:n_x]
            y2 = state[n_x:2 * n_x]
            f1, f2 = self.reaction(y1, y2, parameters)
            dy1 = self.D1 * (laplacian @ y1) + f1
            dy2 = self.D2 * (laplacian @ y2) + f2
            dJ2 = y1[midpoint_index] ** 2 + y2[midpoint_index] ** 2
            return np.r_[dy1, dy2, dJ2]

        solution = solve_ivp(
            rhs,
            (self.t0, self.te),
            initial_state,
            method="LSODA",
            rtol=1e-5,
            atol=1e-7,
        )
        if not solution.success:
            raise RuntimeError(f"fast PDE solve failed: {solution.message}")
        final_state = solution.y[:, -1]
        final_energy = final_state[:n_x] ** 2 + final_state[n_x:2 * n_x] ** 2
        J1 = float(trapezoid(final_energy, x))
        return J1, float(final_state[-1])

    def _cache_metadata(self) -> dict:
        return {
            "version": 2,
            "problem": self.name,
            "pde_nx": self.pde_nx,
            "param_indices": self.param_indices,
            "mass": self.mass,
        }


class HeatProblem(Problem):
    """Deprecated placeholder; superseded by :class:`HeatAMIConProblem`.

    Kept in the codebase for backward compatibility with existing diagnostics
    and tests, but no longer used in ``run_experiments.py``.
    """

    name = "heat_1d_piecewise"
    param_names = ["mu1", "mu2"]
    bounds = np.array([[0.1, 10.0], [0.1, 10.0]])
    qoi_names = ["q"]

    def __init__(self, n_nodes: int = 101) -> None:
        self.n_nodes = int(n_nodes)
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")
        self.x = np.linspace(0.0, 1.0, self.n_nodes)
        self.dx = self.x[1] - self.x[0]
        mass_matrix, _stiffness_matrix = fem.assemble_mass_stiffness(
            self.n_nodes,
            self.dx,
        )
        self.load_vector = fem.lump(mass_matrix)
        self.n_solver_calls = 0

    def solve(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve the finite-element heat system and return x, u."""
        parameters = np.asarray(mu, dtype=float).ravel()
        if parameters.shape != (2,):
            raise ValueError("mu has wrong dimension")
        stiffness_matrix = fem.assemble_variable_coefficient_stiffness(
            self.n_nodes,
            self.dx,
            self._kappa_at_midpoints(parameters),
        )
        system_matrix, system_rhs = fem.apply_dirichlet(
            stiffness_matrix,
            self.load_vector,
            dirichlet_nodes=[0, self.n_nodes - 1],
            values=[0.0, 0.0],
        )
        solution = np.linalg.solve(system_matrix, system_rhs)
        return self.x.copy(), solution

    def evaluate(self, mu: np.ndarray) -> dict[str, float]:
        """Evaluate q = integral_0^1 u(x; mu) dx."""
        _x, solution = self.solve(mu)
        self.n_solver_calls += 1
        return {"q": float(self.load_vector @ solution)}

    def _kappa_at_midpoints(self, parameters: np.ndarray) -> np.ndarray:
        midpoints = self.x[:-1] + 0.5 * self.dx
        return np.where(midpoints < 0.5, parameters[0], parameters[1])


class HeatAMIConProblem(Problem):
    """Deprecated: P1-FEM version. Use HeatAMICon_FD for production (it
    reproduces AMICon-2026 paper exactly via FD discretization with
    harmonic-averaged kappa). Kept for compatibility with existing
    diagnostics.

    PDE: -(kappa(x; mu) u')' = 1 on (0, 1), u(0) = u(1) = 0,
    with kappa = mu_1 if x < 1/2 else mu_2.
    QoI: q(mu) = integral_0^1 u(x; mu) dx.
    """

    name = "heat_amicon_2026"
    param_names = ["mu1", "mu2"]
    bounds = np.array([[0.1, 10.0], [0.1, 10.0]])
    qoi_names = ["q"]

    def __init__(self, n_nodes: int = 51) -> None:
        self.n_nodes = int(n_nodes)
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")
        self.x = np.linspace(0.0, 1.0, self.n_nodes)
        self.dx = self.x[1] - self.x[0]
        mass_matrix, _stiffness_matrix = fem.assemble_mass_stiffness(
            self.n_nodes,
            self.dx,
        )
        self.load_vector = fem.lump(mass_matrix)
        self.n_solver_calls = 0
        self._eval_cache: dict[tuple[float, ...], dict[str, float]] = {}

    def solve(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve the FEM heat system and return (x, u)."""
        parameters = np.asarray(mu, dtype=float).ravel()
        if parameters.shape != (2,):
            raise ValueError("mu has wrong dimension")
        stiffness_matrix = fem.assemble_variable_coefficient_stiffness(
            self.n_nodes,
            self.dx,
            self._kappa_at_midpoints(parameters),
        )
        system_matrix, system_rhs = fem.apply_dirichlet(
            stiffness_matrix,
            self.load_vector,
            dirichlet_nodes=[0, self.n_nodes - 1],
            values=[0.0, 0.0],
        )
        solution = np.linalg.solve(system_matrix, system_rhs)
        return self.x.copy(), solution

    def evaluate(self, mu: np.ndarray) -> dict[str, float]:
        """Evaluate q = integral_0^1 u(x; mu) dx."""
        key = tuple(np.round(np.asarray(mu, dtype=float).ravel(), 10))
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached
        _x, solution = self.solve(mu)
        result = {"q": float(self.load_vector @ solution)}
        self._eval_cache[key] = result
        self.n_solver_calls += 1
        return result

    def _kappa_at_midpoints(self, parameters: np.ndarray) -> np.ndarray:
        midpoints = self.x[:-1] + 0.5 * self.dx
        return np.where(midpoints < 0.5, parameters[0], parameters[1])


class AdvDiffAMIConProblem(Problem):
    """Deprecated: P1-FEM version. Use AdvDiffAMICon_FD for production
    (FD with upwind for advection, stable at high Peclet, reproduces
    AMICon-2026 paper exactly). Kept for compatibility with existing
    diagnostics.

    PDE: -nu u'' + a u' = 1 on (0, 1), u(0) = u(1) = 0.
    QoI: q(mu) = integral_0^1 u(x; mu) dx, mu = (nu, a).

    Standard Galerkin P1 (no SUPG); spurious oscillations may appear at
    high Peclet.
    """

    name = "adv_diff_amicon_2026"
    param_names = ["nu", "a"]
    bounds = np.array([[0.01, 1.0], [0.0, 5.0]])
    qoi_names = ["q"]

    def __init__(self, n_nodes: int = 51) -> None:
        self.n_nodes = int(n_nodes)
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")
        self.x = np.linspace(0.0, 1.0, self.n_nodes)
        self.dx = self.x[1] - self.x[0]
        mass_matrix, stiffness_matrix = fem.assemble_mass_stiffness(
            self.n_nodes,
            self.dx,
        )
        self._stiffness = stiffness_matrix
        self._convection = fem.assemble_convection_p1(self.n_nodes, self.dx)
        self.load_vector = fem.lump(mass_matrix)
        self.n_solver_calls = 0
        self._eval_cache: dict[tuple[float, ...], dict[str, float]] = {}

    def solve(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve the FEM advection-diffusion system and return (x, u)."""
        parameters = np.asarray(mu, dtype=float).ravel()
        if parameters.shape != (2,):
            raise ValueError("mu has wrong dimension")
        nu, advection = float(parameters[0]), float(parameters[1])
        operator = nu * self._stiffness + advection * self._convection
        system_matrix, system_rhs = fem.apply_dirichlet(
            operator,
            self.load_vector,
            dirichlet_nodes=[0, self.n_nodes - 1],
            values=[0.0, 0.0],
        )
        solution = np.linalg.solve(system_matrix, system_rhs)
        return self.x.copy(), solution

    def evaluate(self, mu: np.ndarray) -> dict[str, float]:
        """Evaluate q = integral_0^1 u(x; mu) dx."""
        key = tuple(np.round(np.asarray(mu, dtype=float).ravel(), 10))
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached
        _x, solution = self.solve(mu)
        result = {"q": float(self.load_vector @ solution)}
        self._eval_cache[key] = result
        self.n_solver_calls += 1
        return result


class HeatAMICon_FD(Problem):
    """Steady 1D heat with piecewise-constant kappa - AMICon-2026 reference.

    PDE: -(kappa(x; mu) u'(x))' = 1, u(0) = u(1) = 0,
         kappa = mu_1 for x < 0.5, kappa = mu_2 for x >= 0.5.
    QoI: q(mu) = integral_0^1 u(x; mu) dx.

    Discretization: uniform-grid central FD. kappa is evaluated
    pointwise at the cell half-points x_{i+1/2}. For odd `n_grid`
    (default 51) no half-point lands exactly on the discontinuity at
    x=0.5, so pointwise evaluation gives the same kappa as the textbook
    harmonic-average rule. Reproduces AMICon-2026 paper exactly.

    Reference: O. Shchepanchuk, "Adaptive Sample Selection for Gaussian
    RBF Surrogates of Parametric PDEs" (AMICon-2026).
    """

    name = "heat_amicon_fd"
    param_names = ["mu1", "mu2"]
    bounds = np.array([[0.1, 10.0], [0.1, 10.0]])
    qoi_names = ["q"]

    def __init__(self, n_grid: int = 51) -> None:
        self.n_grid = int(n_grid)
        if self.n_grid < 3:
            raise ValueError("n_grid must be at least 3")
        self.x = np.linspace(0.0, 1.0, self.n_grid)
        self.h = self.x[1] - self.x[0]
        self.x_half = 0.5 * (self.x[:-1] + self.x[1:])
        self._eval_cache: dict[tuple[float, ...], dict[str, float]] = {}
        self.n_solver_calls = 0

    def _kappa(self, x_eval: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.where(x_eval < 0.5, mu[0], mu[1])

    def solve(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from scipy.linalg import solve_banded

        mu = np.asarray(mu, dtype=float).ravel()
        if mu.shape != (2,):
            raise ValueError("mu has wrong dimension")
        n = self.n_grid
        h = self.h
        kh = self._kappa(self.x_half, mu)

        n_int = n - 2
        diag = np.empty(n_int)
        lower = np.zeros(n_int)
        upper = np.zeros(n_int)
        rhs = np.full(n_int, -h * h)

        for j in range(n_int):
            i = j + 1
            k_left = kh[i - 1]
            k_right = kh[i]
            diag[j] = -(k_left + k_right)
            if j > 0:
                lower[j] = k_left
            if j < n_int - 1:
                upper[j] = k_right

        ab = np.zeros((3, n_int))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]
        u_int = solve_banded((1, 1), ab, rhs)

        u = np.zeros(n)
        u[1:-1] = u_int
        return self.x.copy(), u

    def evaluate(self, mu: np.ndarray) -> dict[str, float]:
        key = tuple(np.round(np.asarray(mu, dtype=float).ravel(), 10))
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached
        _x, u = self.solve(mu)
        q = float(np.trapezoid(u, self.x))
        result = {"q": q}
        self._eval_cache[key] = result
        self.n_solver_calls += 1
        return result


class AdvDiffAMICon_FD(Problem):
    """Steady 1D advection-diffusion - AMICon-2026 reference.

    PDE: -nu u''(x) + a u'(x) = 1, u(0) = u(1) = 0.
    QoI: q(mu) = integral_0^1 u(x; mu) dx, mu = (nu, a).

    Discretization: uniform-grid central FD for diffusion, upwind FD
    (backward difference, a >= 0) for advection. Stable at high Peclet.

    Reference: O. Shchepanchuk, AMICon-2026.
    """

    name = "adv_diff_amicon_fd"
    param_names = ["nu", "a"]
    bounds = np.array([[0.01, 1.0], [0.0, 5.0]])
    qoi_names = ["q"]

    def __init__(self, n_grid: int = 51) -> None:
        self.n_grid = int(n_grid)
        if self.n_grid < 3:
            raise ValueError("n_grid must be at least 3")
        self.x = np.linspace(0.0, 1.0, self.n_grid)
        self.h = self.x[1] - self.x[0]
        self._eval_cache: dict[tuple[float, ...], dict[str, float]] = {}
        self.n_solver_calls = 0

    def solve(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from scipy.linalg import solve_banded

        mu = np.asarray(mu, dtype=float).ravel()
        if mu.shape != (2,):
            raise ValueError("mu has wrong dimension")
        nu, a = float(mu[0]), float(mu[1])
        n = self.n_grid
        h = self.h
        n_int = n - 2

        lower_val = -nu - a * h
        diag_val = 2.0 * nu + a * h
        upper_val = -nu
        rhs_val = h * h

        diag = np.full(n_int, diag_val)
        lower = np.full(n_int, lower_val)
        upper = np.full(n_int, upper_val)
        rhs = np.full(n_int, rhs_val)

        ab = np.zeros((3, n_int))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]
        u_int = solve_banded((1, 1), ab, rhs)

        u = np.zeros(n)
        u[1:-1] = u_int
        return self.x.copy(), u

    def evaluate(self, mu: np.ndarray) -> dict[str, float]:
        key = tuple(np.round(np.asarray(mu, dtype=float).ravel(), 10))
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached
        _x, u = self.solve(mu)
        q = float(np.trapezoid(u, self.x))
        result = {"q": q}
        self._eval_cache[key] = result
        self.n_solver_calls += 1
        return result


class BraninProblem(Problem):
    """Branin-Hoo analytic test function (no PDE solve).

    f(x1, x2) = a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*cos(x1) + s,
    with the canonical Branin constants. Three global minima at
    (-pi, 12.275), (pi, 2.275), (9.42478, 2.475) with f* approx 0.397887
    (Forrester-Sobester-Keane 2008, Wiley).
    """

    name = "branin_hoo"
    param_names = ["x1", "x2"]
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
    qoi_names = ["f"]

    def __init__(self) -> None:
        self.n_solver_calls = 0
        self._eval_cache: dict[tuple[float, ...], dict[str, float]] = {}

    def evaluate(self, mu: np.ndarray) -> dict[str, float]:
        """Evaluate the Branin-Hoo functional."""
        parameters = np.asarray(mu, dtype=float).ravel()
        if parameters.shape != (2,):
            raise ValueError("mu has wrong dimension")
        key = tuple(np.round(parameters, 10))
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached
        x1, x2 = float(parameters[0]), float(parameters[1])
        a, b, c, r, s, t = (
            1.0,
            5.1 / (4.0 * math.pi ** 2),
            5.0 / math.pi,
            6.0,
            10.0,
            1.0 / (8.0 * math.pi),
        )
        value = (
            a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
            + s * (1.0 - t) * math.cos(x1)
            + s
        )
        result = {"f": float(value)}
        self._eval_cache[key] = result
        self.n_solver_calls += 1
        return result


def _classify_eigenvalues(eigenvalues: np.ndarray) -> str:
    real_parts = eigenvalues.real
    imaginary_parts = eigenvalues.imag
    if np.any(np.abs(imaginary_parts) > 1e-9):
        if np.allclose(real_parts, 0.0, atol=1e-9):
            return "centre"
        return "stable focus" if np.all(real_parts < 0.0) else "unstable focus"
    if np.all(real_parts < 0.0):
        return "stable node"
    if np.all(real_parts > 0.0):
        return "unstable node"
    return "saddle"


def _self_test() -> None:
    reaction_problem = ReactionDiffusionProblem(pde_nx=21)
    print(
        f"[problems] {reaction_problem.name}: dim={reaction_problem.dim}, "
        f"params={reaction_problem.param_names}, QoIs={reaction_problem.qoi_names}"
    )
    for row in reaction_problem.equilibrium_analysis():
        print(
            f"  equilibrium ({row['y1']:.0f},{row['y2']:.0f}) -> {row['type']}, "
            f"lambda = {row['lambda1']:.3g}, {row['lambda2']:.3g}"
        )

    J = reaction_problem.ode_functional_J()
    x, t, solution = reaction_problem.solve_pde(n_x=161, n_t=201)
    J1, J2 = reaction_problem.pde_functionals_from_solution(x, t, solution)
    print(f"  base functionals: J={J:.3f}, J1={J1:.3f}, J2={J2:.3f}")
    assert 2000.0 < J < 3500.0 and 5.0 < J1 < 40.0 and 3000.0 < J2 < 6000.0

    active_base = reaction_problem.p_base[list(reaction_problem.active)]
    qoi = reaction_problem.evaluate(active_base)
    print(
        f"  evaluate() at base -> {{'J':{qoi['J']:.1f}, "
        f"'J1':{qoi['J1']:.1f}, 'J2':{qoi['J2']:.1f}}}"
    )

    heat_problem = HeatProblem(n_nodes=21)
    heat_qoi = heat_problem.evaluate(np.array([1.0, 1.0]))["q"]
    assert abs(heat_qoi - 1.0 / 12.0) < 1e-3
    print("[problems] self-test OK")


if __name__ == "__main__":
    _self_test()
