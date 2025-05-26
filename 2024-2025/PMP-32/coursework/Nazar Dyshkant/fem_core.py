import numpy as np

def shape_functions_Q4(xi, eta):
    """Returns the shape functions for Q4 element at natural coordinates (xi, eta)."""
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])
    return N

def dN_dxi_Q4(xi, eta):
    """Returns derivatives of Q4 shape functions w.r.t (xi, eta)."""
    dN_dxi = 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],
        [ (1 - eta), -(1 + xi)],
        [ (1 + eta),  (1 + xi)],
        [-(1 + eta),  (1 - xi)]
    ])
    return dN_dxi.T  # shape: (2, 4)

def element_stiffness_Q4(node_coords, D):
    """
    Compute the local stiffness matrix for a Q4 element.
    node_coords: (4, 2) coordinates of the element's nodes
    D: (3, 3) constitutive matrix
    """
    ke = np.zeros((8, 8))

    # Gauss points and weights for 2x2 integration
    gauss_pts = [(-1 / np.sqrt(3), -1 / np.sqrt(3)),
                 ( 1 / np.sqrt(3), -1 / np.sqrt(3)),
                 ( 1 / np.sqrt(3),  1 / np.sqrt(3)),
                 (-1 / np.sqrt(3),  1 / np.sqrt(3))]

    for xi, eta in gauss_pts:
        dN_dxi = dN_dxi_Q4(xi, eta)  # shape: (2, 4)
        J = dN_dxi @ node_coords  # (2, 2)
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        dN_dx = invJ @ dN_dxi  # shape: (2, 4)

        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2 * i]     = dN_dx[0, i]
            B[1, 2 * i + 1] = dN_dx[1, i]
            B[2, 2 * i]     = dN_dx[1, i]
            B[2, 2 * i + 1] = dN_dx[0, i]

        ke += B.T @ D @ B * detJ

    return ke
