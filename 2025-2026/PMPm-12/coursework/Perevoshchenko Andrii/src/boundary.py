import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

from .quadrature import gauss_1d


def apply_dirichlet(A, b, bc_nodes, bc_values):
    A = A.tolil()
    bc_nodes = np.asarray(bc_nodes)
    bc_values = np.asarray(bc_values)

    for k, node in enumerate(bc_nodes):
        val = bc_values[k]
        col = np.array(A[:, node].todense()).ravel()
        b -= col * val
        A[node, :] = 0
        A[:, node] = 0
        A[node, node] = 1.0
        b[node] = val

    return A.tocsc(), b


def apply_dirichlet_simple(A, b, bc_nodes, bc_values):
    A = A.tolil()
    bc_nodes = np.asarray(bc_nodes)
    bc_values = np.asarray(bc_values)

    for k, node in enumerate(bc_nodes):
        A[node, :] = 0
        A[node, node] = 1.0
        b[node] = bc_values[k]

    return A.tocsc(), b


def get_dirichlet_nodes_and_values(mesh, g_func, t=0.0, sides=None):
    if sides is None:
        sides = ['left', 'right', 'bottom', 'top']

    node_set = set()
    for side in sides:
        edges = mesh.boundary_edges.get(side, np.array([], dtype=int))
        if len(edges) > 0:
            node_set.update(edges.ravel())

    bc_nodes = np.array(sorted(node_set), dtype=int)
    coords = mesh.nodes[bc_nodes]
    bc_values = np.array([g_func(x, y, t) for x, y in coords])
    return bc_nodes, bc_values


def assemble_neumann_vector(mesh, q_func, t=0.0, sides=None, n_gauss=2):
    if sides is None:
        sides = []

    n = mesh.n_nodes
    F_N = np.zeros(n)
    qpts, qwts = gauss_1d(n_gauss)

    for side in sides:
        edges = mesh.boundary_edges.get(side, np.array([], dtype=int))
        for edge in edges:
            n1, n2 = edge
            x1, y1 = mesh.nodes[n1]
            x2, y2 = mesh.nodes[n2]
            edge_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            for q_idx in range(len(qwts)):
                s = qpts[q_idx]
                w = qwts[q_idx]
                xq = x1 + s * (x2 - x1)
                yq = y1 + s * (y2 - y1)
                q_val = q_func(xq, yq, t)
                F_N[n1] += w * (1.0 - s) * q_val * edge_len
                F_N[n2] += w * s * q_val * edge_len

    return F_N


def assemble_robin(mesh, alpha, beta_func, t=0.0, sides=None, n_gauss=2):
    if sides is None:
        sides = []

    n = mesh.n_nodes
    R_mat = lil_matrix((n, n))
    R_vec = np.zeros(n)
    qpts, qwts = gauss_1d(n_gauss)

    for side in sides:
        edges = mesh.boundary_edges.get(side, np.array([], dtype=int))
        for edge in edges:
            n1, n2 = edge
            x1, y1 = mesh.nodes[n1]
            x2, y2 = mesh.nodes[n2]
            edge_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            for q_idx in range(len(qwts)):
                s = qpts[q_idx]
                w = qwts[q_idx]
                xq = x1 + s * (x2 - x1)
                yq = y1 + s * (y2 - y1)

                N = np.array([1.0 - s, s])
                b_val = beta_func(xq, yq, t)

                local_ids = [n1, n2]
                for i in range(2):
                    R_vec[local_ids[i]] += w * N[i] * b_val * edge_len
                    for j in range(2):
                        R_mat[local_ids[i], local_ids[j]] += (
                            w * alpha * N[i] * N[j] * edge_len
                        )

    return csc_matrix(R_mat), R_vec
