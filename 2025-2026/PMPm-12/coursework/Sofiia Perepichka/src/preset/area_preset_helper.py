from __future__ import annotations

import numpy as np


class AreaPresetHelper:
    @staticmethod
    def generate_rectangle_boundary(L: float = 1.0, Y: float = 5.0) -> np.ndarray:
        return np.array([
            [0.0, 0.0],
            [L, 0.0],
            [L, Y],
            [0.0, Y],
        ])
