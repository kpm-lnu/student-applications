import numpy as np
from numpy.polynomial.legendre import leggauss
from collections import namedtuple

GaussPoint = namedtuple('GaussPoint', ['xi', 'eta', 'weight'])


class AxisymmetricQuadrature:
    def __init__(self, n_points=None, custom_points=None, custom_weights=None):
        if custom_points is not None and custom_weights is not None:
            self.points = [
                GaussPoint(xi=p[0], eta=p[1], weight=w)
                for p, w in zip(custom_points, custom_weights)
            ]
        else:
            gp_val = 1.0 / np.sqrt(3)
            self.points = [
                GaussPoint(xi=-gp_val, eta=-gp_val, weight=1.0),
                GaussPoint(xi=gp_val, eta=-gp_val, weight=1.0),
                GaussPoint(xi=gp_val, eta=gp_val, weight=1.0),
                GaussPoint(xi=-gp_val, eta=gp_val, weight=1.0),
            ]
    
    def gauss_points(self):
        return self.points