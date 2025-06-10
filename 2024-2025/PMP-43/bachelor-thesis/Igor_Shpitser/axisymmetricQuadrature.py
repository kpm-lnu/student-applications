import numpy as np
from numpy.polynomial.legendre import leggauss


class AxisymmetricQuadrature:
    def __init__(self, n: int | None = None, *, n_points: int | None = None):

        if n is None and n_points is not None:
            n = n_points
        if n is None:
            n = 2

        if n < 1:
            raise ValueError("кількість точок має бути >= 1")

        self.n = int(n)
        self._pts, self._wts = leggauss(self.n)

    def gauss_points(self):
        for i, xi in enumerate(self._pts):
            wi = self._wts[i]
            for j, eta in enumerate(self._pts):
                wj = self._wts[j]
                yield {"xi": xi, "eta": eta, "weight": wi * wj}
