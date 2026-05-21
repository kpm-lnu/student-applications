import numpy as np


def l2_error(mesh, u_num, u_exact_func, t=0.0):
    err_sq = 0.0
    for e in range(mesh.n_elements):
        coords = mesh.element_nodes(e)
        node_ids = mesh.elements[e]
        area = mesh.element_area(e)

        xm = np.mean(coords[:, 0])
        ym = np.mean(coords[:, 1])

        u_h = np.mean(u_num[node_ids])
        u_ex = u_exact_func(xm, ym, t)

        err_sq += (u_h - u_ex) ** 2 * area

    return np.sqrt(err_sq)


def linf_error(mesh, u_num, u_exact_func, t=0.0):
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    u_exact = np.array([u_exact_func(xi, yi, t) for xi, yi in zip(x, y)])
    return np.max(np.abs(u_num - u_exact))


def h1_seminorm_error(mesh, u_num, grad_exact_func, t=0.0):
    from .basis import shape_gradients_physical

    err_sq = 0.0
    for e in range(mesh.n_elements):
        coords = mesh.element_nodes(e)
        node_ids = mesh.elements[e]
        area = mesh.element_area(e)
        grad_phys, _ = shape_gradients_physical(coords)

        u_nodes = u_num[node_ids]
        grad_u_h = grad_phys.T @ u_nodes

        xm = np.mean(coords[:, 0])
        ym = np.mean(coords[:, 1])
        grad_exact = np.array(grad_exact_func(xm, ym, t))

        diff = grad_u_h - grad_exact
        err_sq += np.dot(diff, diff) * area

    return np.sqrt(err_sq)


def compute_peclet_number(velocity, h, D):
    a_norm = np.linalg.norm(velocity)
    if D < 1e-14:
        return np.inf
    return a_norm * h / (2.0 * D)
