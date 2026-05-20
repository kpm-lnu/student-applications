from __future__ import annotations

import numpy as np


class TriangulationService:
    @staticmethod
    def generate_fractional_mesh(
        L: float,
        Y: float,
        N: int,
        M: int,
        gamma: float,
    ) -> dict:
        x_values = np.linspace(0.0, L, N + 1)
        y_values = Y * (np.arange(M + 1) / M) ** gamma

        x_grid, y_grid = np.meshgrid(x_values, y_values)
        vertices = np.column_stack([x_grid.ravel(), y_grid.ravel()])

        triangles = []
        row_size = N + 1

        for j in range(M):
            for i in range(N):
                v1 = j * row_size + i
                v2 = v1 + 1
                v3 = v1 + row_size
                v4 = v3 + 1

                triangles.append([v1, v2, v4])
                triangles.append([v1, v4, v3])

        return {
            "vertices": vertices,
            "triangles": np.array(triangles),
        }
