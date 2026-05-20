from __future__ import annotations

import numpy as np


class HelperFunctions:
    @staticmethod
    def compute_delta(triangle_vertices: np.ndarray) -> float:
        x1, y1 = triangle_vertices[0]
        x2, y2 = triangle_vertices[1]
        x3, y3 = triangle_vertices[2]

        area = 0.5 * (
            (x1 * y2 + x2 * y3 + x3 * y1)
            - (y1 * x2 + y2 * x3 + y3 * x1)
        )
        return 2.0 * area

    @staticmethod
    def compute_boundary_length(vertex1: np.ndarray, vertex2: np.ndarray) -> float:
        return float(np.linalg.norm(np.array(vertex1) - np.array(vertex2)))

    @staticmethod
    def get_boundary_edges(triangles: np.ndarray) -> list[tuple[int, int]]:
        edge_counts: dict[tuple[int, int], int] = {}

        for triangle in triangles:
            edges = [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ]

            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                edge_counts[sorted_edge] = edge_counts.get(sorted_edge, 0) + 1

        return [edge for edge, count in edge_counts.items() if count == 1]
