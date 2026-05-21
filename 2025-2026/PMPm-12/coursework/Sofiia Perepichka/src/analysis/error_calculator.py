from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from src.analysis.trace_solution import TraceSolution
from src.problem.fractional_problem import FractionalProblem


@dataclass(slots=True)
class ErrorCalculator:
    problem: FractionalProblem

    gauss_points: ClassVar[np.ndarray] = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
    gauss_weights: ClassVar[np.ndarray] = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    def compute_l2_error(self, vertices: np.ndarray, values: np.ndarray) -> float:
        return self._integrate_trace_error(vertices, values, include_derivative=False)

    def compute_w21_error(self, vertices: np.ndarray, values: np.ndarray) -> float:
        return self._integrate_trace_error(vertices, values, include_derivative=True)

    def _integrate_trace_error(
        self,
        vertices: np.ndarray,
        values: np.ndarray,
        include_derivative: bool,
    ) -> float:
        x_values, u_values = TraceSolution.extract(vertices, values)
        error_squared = 0.0

        for index in range(len(x_values) - 1):
            x_left = x_values[index]
            x_right = x_values[index + 1]
            element_length = x_right - x_left

            if element_length <= 0:
                continue

            u_left = u_values[index]
            u_right = u_values[index + 1]
            slope = (u_right - u_left) / element_length
            midpoint = 0.5 * (x_left + x_right)
            half_length = 0.5 * element_length
            quadrature_points = midpoint + half_length * self.gauss_points
            exact_values = self.problem.exact_solution(quadrature_points)
            fem_values = u_left + slope * (quadrature_points - x_left)
            local_error = (exact_values - fem_values) ** 2

            if include_derivative:
                exact_derivative = self.problem.exact_solution_derivative(quadrature_points)
                local_error = local_error + (exact_derivative - slope) ** 2

            error_squared += half_length * np.sum(self.gauss_weights * local_error)

        return float(np.sqrt(error_squared))
