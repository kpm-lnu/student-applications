"""
Матеріал для осесиметричної задачі (плоский стан деформації).
"""

import numpy as np


class Material:
    def __init__(self, name: str, E: float, nu: float):
        self.name = name
        self.E = float(E)
        self.nu = float(nu)

    def get_elastic_matrix(self) -> np.ndarray:
        """
        Осьовосиметрична матриця C (4 × 4) у порядку
        [σ_rr, σ_zz, σ_rz, σ_θθ].
        """
        E, nu = self.E, self.nu
        coef = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        C11 = (1.0 - nu) * coef
        C12 = nu * coef
        C33 = (1.0 - 2.0 * nu) / 2.0 * coef

        return np.array(
            [
                [C11, C12, 0.0, C12],
                [C12, C11, 0.0, C12],
                [0.0, 0.0, C33, 0.0],
                [C12, C12, 0.0, C11],
            ],
            dtype=float,
        )
