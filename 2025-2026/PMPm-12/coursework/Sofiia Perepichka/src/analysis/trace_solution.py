from __future__ import annotations

import numpy as np


class TraceSolution:
    @staticmethod
    def extract(vertices: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bottom_nodes = np.where(np.abs(vertices[:, 1]) < 1e-12)[0]
        x_values = vertices[bottom_nodes, 0]
        u_values = values[bottom_nodes]
        order = np.argsort(x_values)
        return x_values[order], u_values[order]
