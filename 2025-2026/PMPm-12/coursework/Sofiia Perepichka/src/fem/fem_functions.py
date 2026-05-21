from __future__ import annotations

from typing import Callable

import numpy as np

from src.common.helpers import HelperFunctions


class FemService:
    @staticmethod
    def compute_stiffness_matrix(
        triangle_vertices: np.ndarray,
        a11: float,
        a22: float,
        alpha: float,
    ) -> np.ndarray:
        (x1, y1), (x2, y2), (x3, y3) = triangle_vertices
        delta = abs(HelperFunctions.compute_delta(triangle_vertices))

        if delta == 0:
            return np.zeros((3, 3))

        barycentric_points = np.array([
            [2 / 3, 1 / 6, 1 / 6],
            [1 / 6, 2 / 3, 1 / 6],
            [1 / 6, 1 / 6, 2 / 3],
        ])
        quadrature_points = FemService._compute_quadrature_points(triangle_vertices, barycentric_points)
        weight = (1.0 / 3.0) * np.sum(quadrature_points[:, 1] ** alpha)
        x_diff_12 = x1 - x2
        x_diff_23 = x2 - x3
        x_diff_31 = x3 - x1
        y_diff_12 = y1 - y2
        y_diff_23 = y2 - y3
        y_diff_31 = y3 - y1
        base_matrix = (1.0 / (2.0 * delta)) * np.array([
            [
                a11 * y_diff_23**2 + a22 * x_diff_23**2,
                a11 * y_diff_23 * y_diff_31 + a22 * x_diff_23 * x_diff_31,
                a11 * y_diff_23 * y_diff_12 + a22 * x_diff_23 * x_diff_12,
            ],
            [
                a11 * y_diff_31 * y_diff_23 + a22 * x_diff_31 * x_diff_23,
                a11 * y_diff_31**2 + a22 * x_diff_31**2,
                a11 * y_diff_31 * y_diff_12 + a22 * x_diff_31 * x_diff_12,
            ],
            [
                a11 * y_diff_12 * y_diff_23 + a22 * x_diff_12 * x_diff_23,
                a11 * y_diff_12 * y_diff_31 + a22 * x_diff_12 * x_diff_31,
                a11 * y_diff_12**2 + a22 * x_diff_12**2,
            ],
        ])
        return weight * base_matrix

    @staticmethod
    def compute_mass_matrix(triangle_vertices: np.ndarray, alpha: float) -> np.ndarray:
        delta = abs(HelperFunctions.compute_delta(triangle_vertices))

        if delta == 0:
            return np.zeros((3, 3))

        area = 0.5 * delta
        barycentric_points = np.array([
            [2 / 3, 1 / 6, 1 / 6],
            [1 / 6, 2 / 3, 1 / 6],
            [1 / 6, 1 / 6, 2 / 3],
        ])
        quadrature_points = FemService._compute_quadrature_points(triangle_vertices, barycentric_points)
        weights = quadrature_points[:, 1] ** alpha
        mass_matrix = np.zeros((3, 3))

        for index in range(3):
            phi = barycentric_points[index]
            mass_matrix += weights[index] * np.outer(phi, phi)

        return mass_matrix * (area / 3.0)

    @staticmethod
    def compute_source_vector(
        edge_vertices: np.ndarray,
        source_function: Callable[[np.ndarray | float], np.ndarray | float],
        ds: float,
    ) -> np.ndarray:
        p1 = edge_vertices[0]
        p2 = edge_vertices[1]
        length = HelperFunctions.compute_boundary_length(p1, p2)
        mass_matrix = (length / 6.0) * np.array([
            [2.0, 1.0],
            [1.0, 2.0],
        ])
        source_values = np.array([source_function(p1[0]), source_function(p2[0])])
        source_vector = ds * (mass_matrix @ source_values)
        return source_vector.reshape(2, 1)

    @staticmethod
    def compute_global_matrix(global_matrix, matrix: np.ndarray, triangle: np.ndarray):
        for i in range(3):
            for j in range(3):
                global_matrix[triangle[i], triangle[j]] += matrix[i, j]

        return global_matrix

    @staticmethod
    def compute_global_vector(local_vector: np.ndarray, edge: np.ndarray, global_vector: np.ndarray):
        for i in range(2):
            global_vector[edge[i]] += local_vector[i]

        return global_vector

    @staticmethod
    def _compute_quadrature_points(
        triangle_vertices: np.ndarray,
        barycentric_points: np.ndarray,
    ) -> np.ndarray:
        v1, v2, v3 = triangle_vertices
        return (
            barycentric_points[:, [0]] * v1
            + barycentric_points[:, [1]] * v2
            + barycentric_points[:, [2]] * v3
        )
