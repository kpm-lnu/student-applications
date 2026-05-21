"""P1 finite elements on a uniform one-dimensional mesh.

On the reference element xi in [0, 1], phi_0 = 1 - xi and phi_1 = xi.
The affine map x = x_e + dx * xi gives
M_e = dx / 6 [[2, 1], [1, 2]] and K_e = 1 / dx [[1, -1], [-1, 1]].
For the parabolic weak form, M u_t + K u = M f, so the nodal diffusion
operator with natural Neumann data is -M^{-1} K. Dirichlet values are
imposed below by symmetric row/column replacement after moving their
contribution into the right-hand side.
"""
from __future__ import annotations

import numpy as np


def assemble_mass_stiffness(n_nodes: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Consistent mass and stiffness matrices for P1 elements."""
    n_nodes = _validate_mesh(n_nodes, dx)
    mass_local = dx / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    stiffness_local = 1.0 / dx * np.array([[1.0, -1.0], [-1.0, 1.0]])

    mass_matrix = np.zeros((n_nodes, n_nodes))
    stiffness_matrix = np.zeros((n_nodes, n_nodes))
    for element in range(n_nodes - 1):
        element_slice = slice(element, element + 2)
        mass_matrix[element_slice, element_slice] += mass_local
        stiffness_matrix[element_slice, element_slice] += stiffness_local
    return mass_matrix, stiffness_matrix


def lump(mass_matrix: np.ndarray) -> np.ndarray:
    """Diagonal of the row-sum lumped mass matrix."""
    mass_matrix = np.asarray(mass_matrix, dtype=float)
    if mass_matrix.ndim != 2 or mass_matrix.shape[0] != mass_matrix.shape[1]:
        raise ValueError("mass matrix must be square")
    return mass_matrix.sum(axis=1)


def laplacian_neumann(
    n_nodes: int,
    dx: float,
    mass: str = "consistent",
) -> np.ndarray:
    """Dense matrix for the FEM operator -M^{-1} K with natural Neumann data."""
    mass_matrix, stiffness_matrix = assemble_mass_stiffness(n_nodes, dx)
    if mass == "lumped":
        return -stiffness_matrix / lump(mass_matrix)[:, None]
    if mass == "consistent":
        return np.linalg.solve(mass_matrix, -stiffness_matrix)
    raise ValueError("mass must be 'consistent' or 'lumped'")


def apply_dirichlet(
    stiffness_matrix: np.ndarray,
    rhs: np.ndarray,
    dirichlet_nodes: np.ndarray | list[int] | tuple[int, ...],
    values: np.ndarray | list[float] | tuple[float, ...] | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Dirichlet values by symmetric row/column replacement."""
    system_matrix = np.array(stiffness_matrix, dtype=float, copy=True)
    system_rhs = np.array(rhs, dtype=float, copy=True).ravel()
    if system_matrix.ndim != 2 or system_matrix.shape[0] != system_matrix.shape[1]:
        raise ValueError("system matrix must be square")
    if system_rhs.shape != (system_matrix.shape[0],):
        raise ValueError("rhs has incompatible shape")

    boundary_nodes = np.asarray(dirichlet_nodes, dtype=int).ravel()
    boundary_values = np.asarray(values, dtype=float)
    if boundary_values.ndim == 0:
        boundary_values = np.full(boundary_nodes.shape, float(boundary_values))
    else:
        boundary_values = boundary_values.ravel()
    if boundary_nodes.shape != boundary_values.shape:
        raise ValueError("Dirichlet nodes and values must match")
    if np.any(boundary_nodes < 0) or np.any(boundary_nodes >= system_matrix.shape[0]):
        raise ValueError("Dirichlet node out of range")

    system_rhs -= system_matrix[:, boundary_nodes] @ boundary_values
    for node, value in zip(boundary_nodes, boundary_values):
        system_matrix[node, :] = 0.0
        system_matrix[:, node] = 0.0
        system_matrix[node, node] = 1.0
        system_rhs[node] = value
    return system_matrix, system_rhs


def assemble_variable_coefficient_stiffness(
    n_nodes: int,
    dx: float,
    kappa_at_midpoints: np.ndarray,
) -> np.ndarray:
    """P1 stiffness matrix for piecewise-constant element coefficients."""
    n_nodes = _validate_mesh(n_nodes, dx)
    coefficients = np.asarray(kappa_at_midpoints, dtype=float).ravel()
    if coefficients.shape != (n_nodes - 1,):
        raise ValueError("coefficient vector has wrong length")

    stiffness_matrix = np.zeros((n_nodes, n_nodes))
    reference_stiffness = np.array([[1.0, -1.0], [-1.0, 1.0]])
    for element, coefficient in enumerate(coefficients):
        element_slice = slice(element, element + 2)
        stiffness_matrix[element_slice, element_slice] += (
            coefficient / dx * reference_stiffness
        )
    return stiffness_matrix


def assemble_convection_p1(n_nodes: int, dx: float) -> np.ndarray:
    """Galerkin convection matrix C_ij = integral phi_i phi_j' dx for P1.

    On the reference element the local convection matrix is the
    asymmetric tridiagonal pattern (1/2)*[[-1, 1], [-1, 1]], independent
    of dx (the Jacobians of phi' and the integration cancel).
    """
    _validate_mesh(n_nodes, dx)
    convection_local = 0.5 * np.array([[-1.0, 1.0], [-1.0, 1.0]])
    convection_matrix = np.zeros((n_nodes, n_nodes))
    for element in range(n_nodes - 1):
        element_slice = slice(element, element + 2)
        convection_matrix[element_slice, element_slice] += convection_local
    return convection_matrix


def _validate_mesh(n_nodes: int, dx: float) -> int:
    n_nodes_int = int(n_nodes)
    if n_nodes_int != n_nodes or n_nodes_int < 2:
        raise ValueError("n_nodes must be at least 2")
    if dx <= 0:
        raise ValueError("dx must be positive")
    return n_nodes_int


def _self_test() -> None:
    n_nodes = 21
    dx = 0.2
    rng = np.random.default_rng(4)
    values = rng.normal(size=n_nodes)
    operator = laplacian_neumann(n_nodes, dx, mass="lumped")

    reference = np.empty_like(values)
    reference[1:-1] = (values[:-2] - 2.0 * values[1:-1] + values[2:]) / dx**2
    reference[0] = 2.0 * (values[1] - values[0]) / dx**2
    reference[-1] = 2.0 * (values[-2] - values[-1]) / dx**2

    np.testing.assert_allclose(operator @ values, reference, atol=1e-12)
    mass_matrix, stiffness_matrix = assemble_mass_stiffness(n_nodes, dx)
    assert mass_matrix.shape == stiffness_matrix.shape == (n_nodes, n_nodes)
    print("[fem] self-test OK")


if __name__ == "__main__":
    _self_test()
