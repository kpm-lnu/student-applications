import numpy as np

GRAD_REF = np.array([
    [-1.0, -1.0],
    [ 1.0,  0.0],
    [ 0.0,  1.0],
])


def shape_functions(xi, eta):
    return np.array([1.0 - xi - eta, xi, eta])


def shape_gradients_ref():
    return GRAD_REF.copy()


def jacobian(coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    J = np.array([
        [x2 - x1, x3 - x1],
        [y2 - y1, y3 - y1],
    ])
    det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    return J, det_J


def shape_gradients_physical(coords):
    J, det_J = jacobian(coords)
    inv_JT = np.array([
        [ J[1, 1], -J[1, 0]],
        [-J[0, 1],  J[0, 0]],
    ]) / det_J

    grad_phys = GRAD_REF @ inv_JT.T
    return grad_phys, det_J
