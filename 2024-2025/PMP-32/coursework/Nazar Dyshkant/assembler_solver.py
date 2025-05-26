import numpy as np
from fem_core import element_stiffness_Q4

def assemble_global_stiffness(nodes, elements, D):
    """
    Assembles the global stiffness matrix for all elements.
    """
    num_nodes = nodes.shape[0]
    K = np.zeros((2 * num_nodes, 2 * num_nodes))

    for element in elements:
        node_indices = np.array(element)
        coords = nodes[node_indices]  # (4, 2)
        ke = element_stiffness_Q4(coords, D)  # (8, 8)

        # Assemble into global K
        dof_indices = []
        for idx in node_indices:
            dof_indices.extend([2 * idx, 2 * idx + 1])

        for i in range(8):
            for j in range(8):
                K[dof_indices[i], dof_indices[j]] += ke[i, j]

    return K

def apply_dirichlet_bc(K, F, fixed_dofs, values=None):
    """
    Apply Dirichlet boundary conditions.
    fixed_dofs: list of DOF indices to fix (e.g., [0, 1, 20, 21, ...])
    values: corresponding displacement values (defaults to 0)
    """
    if values is None:
        values = np.zeros(len(fixed_dofs))

    for dof, val in zip(fixed_dofs, values):
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1
        F[dof] = val

    return K, F


def B_matrix_Q4(coords):
    """
    Returns the strain-displacement matrix B for a Q4 element evaluated at the center (xi=0, eta=0).
    """
    xi = eta = 0  # center of the element
    dN_dxi = 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],
        [(1 - eta), -(1 + xi)],
        [(1 + eta), (1 + xi)],
        [-(1 + eta), (1 - xi)]
    ])  # dN/dxi and dN/deta stacked

    dN_dxi = dN_dxi.reshape((4, 2))
    J = dN_dxi.T @ coords
    detJ = np.linalg.det(J)
    dN_dx = np.linalg.solve(J.T, dN_dxi.T).T

    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2 * i] = dN_dx[i, 0]
        B[1, 2 * i + 1] = dN_dx[i, 1]
        B[2, 2 * i] = dN_dx[i, 1]
        B[2, 2 * i + 1] = dN_dx[i, 0]

    return B, detJ, J


def apply_dirichlet_bc_penalty(K, F, fixed_dofs, penalty=1e15):
    for dof in fixed_dofs:
        K[dof, dof] += penalty
        F[dof] += penalty * 0  # assuming zero displacement
    return K, F


def solve_system(K, F):
    """
    Solve the linear system KU = F
    """
    U = np.linalg.solve(K, F)
    return U
