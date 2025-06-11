import numpy as np


def generate_mesh(nx, ny, domain=(0, 1, 0, 1)):
    """
    Генерація регулярної трикутної сітки на прямокутнику.

    :param nx: кількість поділок по x
    :param ny: кількість поділок по y
    :param domain: (x_min, x_max, y_min, y_max)
    :return: nodes, elements, boundary_nodes
    """
    x_min, x_max, y_min, y_max = domain
    hx = (x_max - x_min) / nx
    hy = (y_max - y_min) / ny

    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = x_min + i * hx
            y = y_min + j * hy
            nodes.append((x, y))

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + nx + 1
            n3 = n2 + 1

            elements.append((n0, n1, n3))
            elements.append((n0, n3, n2))

    boundary_nodes = []
    for idx, (x, y) in enumerate(nodes):
        if np.isclose(x, x_min) or np.isclose(x, x_max) or \
           np.isclose(y, y_min) or np.isclose(y, y_max):
            boundary_nodes.append(idx)

    return nodes, elements, boundary_nodes
