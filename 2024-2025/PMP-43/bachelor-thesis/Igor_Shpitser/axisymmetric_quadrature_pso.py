"""
Квадратура, параметризована алгоритмом PSO.
Кожній точці відповідають три значення: ξ, η та вага w.
"""

import numpy as np


class PSOQuadrature:
    def __init__(self, xi: np.ndarray, eta: np.ndarray, weights: np.ndarray):
        self._xi = np.asarray(xi, dtype=float)
        self._eta = np.asarray(eta, dtype=float)
        self._w = np.asarray(weights, dtype=float)

        if not (self._xi.shape == self._eta.shape == self._w.shape):
            raise ValueError("xi, eta та weights повинні мати однакову довжину")

    # кількість точок
    @property
    def n_points(self) -> int:
        return len(self._xi)

    # генератор словників {"xi": ξ, "eta": η, "weight": w}
    def gauss_points(self):
        for xi, eta, w in zip(self._xi, self._eta, self._w):
            yield {"xi": xi, "eta": eta, "weight": w}
