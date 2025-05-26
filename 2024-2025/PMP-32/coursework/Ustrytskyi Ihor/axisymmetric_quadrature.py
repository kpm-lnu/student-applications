import numpy as np
from numpy.polynomial.legendre import leggauss

class AxisymmetricQuadrature:
    def __init__(self, n_points: int):
        self.n_points = n_points
        
    def gauss_points(self):
        gp_val = 1.0 / np.sqrt(3)
        return [
            {"xi": -gp_val, "eta": -gp_val, "weight": 1.0},
            {"xi":  gp_val, "eta": -gp_val, "weight": 1.0},
            {"xi":  gp_val, "eta":  gp_val, "weight": 1.0},
            {"xi": -gp_val, "eta":  gp_val, "weight": 1.0},
        ]