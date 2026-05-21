import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from .element import local_stiffness, element_dofs


def assemble_stiffness(nodes: np.ndarray, elements: np.ndarray, D: np.ndarray, thickness: float) -> tuple[csr_matrix, list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Формує глобальну матрицю жорсткості.
    Повертає K, список матриць B, список dofs для елементів, площі елементів.
    """
    ndof = 2 * len(nodes)
    K = lil_matrix((ndof, ndof), dtype=float)
    B_list: list[np.ndarray] = []
    dof_list: list[np.ndarray] = []
    areas = []

    for elem in elements:
        coords = nodes[elem]
        Ke, area, B = local_stiffness(coords, D, thickness)
        dofs = element_dofs(elem)

        for i_local, i_global in enumerate(dofs):
            for j_local, j_global in enumerate(dofs):
                K[i_global, j_global] += Ke[i_local, j_local]

        B_list.append(B)
        dof_list.append(dofs)
        areas.append(area)

    return K.tocsr(), B_list, dof_list, np.array(areas, dtype=float)


def add_edge_traction(F: np.ndarray, nodes: np.ndarray, edge: tuple[int, int], traction: np.ndarray, thickness: float) -> None:
    """
    Додає еквівалентні вузлові сили для сталої поверхневої сили на ребрі.
    traction: вектор [t_x, t_y] у Н/м^2.
    """
    a, b = edge
    length = float(np.linalg.norm(nodes[b] - nodes[a]))
    fe_node = traction * thickness * length / 2.0

    F[2 * a:2 * a + 2] += fe_node
    F[2 * b:2 * b + 2] += fe_node


def add_edge_traction_function(F: np.ndarray, nodes: np.ndarray, edge: tuple[int, int], traction_func, thickness: float) -> None:
    """
    Додає еквівалентні вузлові сили для навантаження, заданого функцією від середини ребра.
    """
    a, b = edge
    mid = 0.5 * (nodes[a] + nodes[b])
    traction = np.asarray(traction_func(mid), dtype=float)
    add_edge_traction(F, nodes, edge, traction, thickness)


def global_force_vector(n_nodes: int) -> np.ndarray:
    return np.zeros(2 * n_nodes, dtype=float)
