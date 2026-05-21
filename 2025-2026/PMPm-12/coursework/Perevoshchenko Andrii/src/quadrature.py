import numpy as np


def gauss_triangle_1():
    points = np.array([[1.0 / 3.0, 1.0 / 3.0]])
    weights = np.array([0.5])
    return points, weights


def gauss_triangle_3():
    points = np.array([
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0],
    ])
    weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    return points, weights


def gauss_triangle_4():
    points = np.array([
        [1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 5.0, 1.0 / 5.0],
        [3.0 / 5.0, 1.0 / 5.0],
        [1.0 / 5.0, 3.0 / 5.0],
    ])
    weights = np.array([-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0])
    return points, weights


def get_quadrature(order=2):
    if order <= 1:
        return gauss_triangle_1()
    elif order <= 2:
        return gauss_triangle_3()
    else:
        return gauss_triangle_4()


def gauss_1d(n_points=2):
    if n_points == 1:
        return np.array([0.5]), np.array([1.0])
    elif n_points == 2:
        s = 1.0 / np.sqrt(3.0)
        return np.array([0.5 - 0.5 * s, 0.5 + 0.5 * s]), np.array([0.5, 0.5])
    else:
        s1 = np.sqrt(3.0 / 5.0)
        return (
            np.array([0.5 - 0.5 * s1, 0.5, 0.5 + 0.5 * s1]),
            np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0]),
        )
