import numpy as np


def compute_ke(x1, x2, x3, a11=1.0, a22=1.0):
    """
    Побудова локальної матриці жорсткості Ke для трикутного елемента.
    """
    C = np.array([
        [x2[1] - x3[1], x3[1] - x1[1], x1[1] - x2[1]],
        [x3[0] - x2[0], x1[0] - x3[0], x2[0] - x1[0]]
    ])
    area = 0.5 * abs(np.linalg.det(np.array([
        [1, x1[0], x1[1]],
        [1, x2[0], x2[1]],
        [1, x3[0], x3[1]]
    ])))
    A = np.array([
        [a11, 0],
        [0, a22]
    ])
    Ke = (1.0 / (4.0 * area)) * C.T @ A @ C
    return Ke


def compute_me(area):
    """
    Побудова локальної масової матриці Me для трикутного елемента.
    """
    return (area / 12.0) * np.array([
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ])


def compute_qe(x1, x2, x3, f_value):
    """
    Побудова локального вектора правої частини Qe.
    """
    area = 0.5 * abs(np.linalg.det(np.array([
        [1, x1[0], x1[1]],
        [1, x2[0], x2[1]],
        [1, x3[0], x3[1]]
    ])))
    return f_value * (area / 3.0) * np.ones(3)


def assemble_global_matrix(nodes, elements, a11=1.0, a22=1.0, f_func=None):
    N = len(nodes)
    K = np.zeros((N, N))
    F = np.zeros(N)

    for triangle in elements:
        i, j, k = triangle
        x1, x2, x3 = nodes[i], nodes[j], nodes[k]

        Ke = compute_ke(x1, x2, x3, a11, a22)

        if f_func:
            f_val = (f_func(*x1) + f_func(*x2) + f_func(*x3)) / 3
        else:
            f_val = 1.0

        Qe = compute_qe(x1, x2, x3, f_val)

        for local_i, global_i in enumerate(triangle):
            F[global_i] += Qe[local_i]
            for local_j, global_j in enumerate(triangle):
                K[global_i, global_j] += Ke[local_i, local_j]

    return K, F



def apply_dirichlet(K, F, boundary_nodes, u_D=0.0):
    """
    Накладення умов Діріхле (фіксація значень на межі).
    """
    for idx in boundary_nodes:
        K[idx, :] = 0.0
        K[:, idx] = 0.0
        K[idx, idx] = 1.0
        F[idx] = u_D
    return K, F


def solve_system(K, F):
    """
    Розв’язання системи лінійних алгебраїчних рівнянь K·u = F.
    """
    return np.linalg.solve(K, F)
