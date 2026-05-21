import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

from .basis import shape_functions, shape_gradients_physical
from .quadrature import get_quadrature


def compute_tau_supg(h_e, velocity, D):
    a_norm = np.linalg.norm(velocity)
    if a_norm < 1e-14:
        return 0.0
    Pe = a_norm * h_e / (2.0 * D) if D > 1e-14 else 1e10
    if Pe < 1e-8:
        return h_e ** 2 / (4.0 * D)
    tau = (h_e / (2.0 * a_norm)) * (1.0 / np.tanh(Pe) - 1.0 / Pe)
    return tau


def assemble_supg_matrix(mesh, velocity, D, quad_order=2):
    a = np.asarray(velocity, dtype=float)
    n = mesh.n_nodes
    S = lil_matrix((n, n))
    qpts, qwts = get_quadrature(quad_order)

    for e in range(mesh.n_elements):
        coords = mesh.element_nodes(e)
        node_ids = mesh.elements[e]
        grad_phys, det_J = shape_gradients_physical(coords)
        area_factor = abs(det_J)

        h_e = max(
            np.linalg.norm(coords[i] - coords[j])
            for i in range(3) for j in range(i + 1, 3)
        )

        tau = compute_tau_supg(h_e, a, D)
        if tau < 1e-16:
            continue

        a_dot_grad = grad_phys @ a

        Se = np.zeros((3, 3))
        for q in range(len(qwts)):
            Se += qwts[q] * tau * np.outer(a_dot_grad, a_dot_grad) * area_factor

        for i in range(3):
            for j in range(3):
                S[node_ids[i], node_ids[j]] += Se[i, j]

    return csc_matrix(S)


def assemble_supg_mass_matrix(mesh, velocity, D, quad_order=2):
    a = np.asarray(velocity, dtype=float)
    n = mesh.n_nodes
    SM = lil_matrix((n, n))
    qpts, qwts = get_quadrature(quad_order)

    for e in range(mesh.n_elements):
        coords = mesh.element_nodes(e)
        node_ids = mesh.elements[e]
        grad_phys, det_J = shape_gradients_physical(coords)
        area_factor = abs(det_J)

        h_e = max(
            np.linalg.norm(coords[i] - coords[j])
            for i in range(3) for j in range(i + 1, 3)
        )

        tau = compute_tau_supg(h_e, a, D)
        if tau < 1e-16:
            continue

        a_dot_grad = grad_phys @ a

        SMe = np.zeros((3, 3))
        for q in range(len(qwts)):
            xi, eta = qpts[q]
            N = shape_functions(xi, eta)
            SMe += qwts[q] * tau * np.outer(a_dot_grad, N) * area_factor

        for i in range(3):
            for j in range(3):
                SM[node_ids[i], node_ids[j]] += SMe[i, j]

    return csc_matrix(SM)
