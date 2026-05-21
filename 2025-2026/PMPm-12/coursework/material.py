import numpy as np


def elasticity_matrix(E: float, nu: float, plane_type: str = "stress") -> np.ndarray:
    """
    Формує матрицю пружності D для плоскої задачі.

    plane_type:
        "stress"  - плоский напружений стан;
        "strain"  - плоска деформація.
    """
    if E <= 0:
        raise ValueError("Модуль Юнга E має бути додатним.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("Коефіцієнт Пуассона має бути в межах (-1, 0.5).")

    if plane_type == "stress":
        coef = E / (1.0 - nu ** 2)
        return coef * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ], dtype=float)

    if plane_type == "strain":
        coef = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return coef * np.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
        ], dtype=float)

    raise ValueError("plane_type має бути 'stress' або 'strain'.")
