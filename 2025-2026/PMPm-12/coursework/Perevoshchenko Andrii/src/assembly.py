import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

from .basis import shape_functions, shape_gradients_physical
from .quadrature import get_quadrature


def assemble_mass_matrix(mesh, quad_order=2):
    n = mesh.n_nodes
    M = lil_matrix((n, n))
    qpts, qwts = get_quadrature(quad_order)

    for e in range(mesh.n_elements):
        coords = mesh.element_nodes(e)
        node_ids = mesh.elements[e]
        _, det_J = shape_gradients_physical(coords)
        area_factor = abs(det_J)

        Me = np.zeros((3, 3))
        for q in range(len(qwts)):
            xi, eta = qpts[q]
            N = shape_functions(xi, eta)
            Me += qwts[q] * np.outer(N, N) * area_factor

        for i in range(3):
            for j in range(3):
                M[node_ids[i], node_ids[j]] += Me[i, j]

    return csc_matrix(M)


def assemble_stiffness_matrix(mesh, D, quad_order=2):
    n = mesh.n_nodes
    K = lil_matrix((n, n))
    qpts, qwts = get_quadrature(quad_order)

    for e in range(mesh.n_elements):
        coords = mesh.element_nodes(e)
        node_ids = mesh.elements[e]
        grad_phys, det_J = shape_gradients_physical(coords)
        area_factor = abs(det_J)

        Ke = np.zeros((3, 3))
        for q in range(len(qwts)):
            Ke += qwts[q] * (grad_phys @ grad_phys.T) * D * area_factor

        for i in range(3):
            for j in range(3):
                K[node_ids[i], node_ids[j]] += Ke[i, j]

    return csc_matrix(K)


def assemble_advection_matrix(mesh, velocity, quad_order=2):
    a = np.asarray(velocity, dtype=float)
    n = mesh.n_nodes
    C = lil_matrix((n, n))
    qpts, qwts = get_quadrature(quad_order)

    for e in range(mesh.n_elements):
        coords = mesh.element_nodes(e)
        node_ids = mesh.elements[e]
        grad_phys, det_J = shape_gradients_physical(coords)
        area_factor = abs(det_J)

        a_dot_grad = grad_phys @ a

        Ce = np.zeros((3, 3))
        for q in range(len(qwts)):
            xi, eta = qpts[q]
            N = shape_functions(xi, eta)
            Ce += qwts[q] * np.outer(N, a_dot_grad) * area_factor

        for i in range(3):
            for j in range(3):
                C[node_ids[i], node_ids[j]] += Ce[i, j]

    return csc_matrix(C)


def assemble_load_vector(mesh, f_func, t=0.0, quad_order=2):
    n = mesh.n_nodes
    F = np.zeros(n)
    qpts, qwts = get_quadrature(quad_order)

    for e in range(mesh.n_elements):
        coords = mesh.element_nodes(e)
        node_ids = mesh.elements[e]
        _, det_J = shape_gradients_physical(coords)
        area_factor = abs(det_J)

        Fe = np.zeros(3)
        for q in range(len(qwts)):
            xi, eta = qpts[q]
            N = shape_functions(xi, eta)
            x_phys = coords[:, 0] @ N
            y_phys = coords[:, 1] @ N
            fval = f_func(x_phys, y_phys, t)
            Fe += qwts[q] * N * fval * area_factor

        for i in range(3):
            F[node_ids[i]] += Fe[i]

    return F


def assemble_all(mesh, D, velocity, f_func=None, t=0.0, quad_order=2):
    M = assemble_mass_matrix(mesh, quad_order)
    K = assemble_stiffness_matrix(mesh, D, quad_order)
    C = assemble_advection_matrix(mesh, velocity, quad_order)
    if f_func is not None:
        F = assemble_load_vector(mesh, f_func, t, quad_order)
    else:
        F = np.zeros(mesh.n_nodes)
    return M, K, C, F
