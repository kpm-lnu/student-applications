from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from src.analysis.trace_solution import TraceSolution
from src.problem.fractional_problem import FractionalProblem
from src.solver.solution_result import SolutionResult


class PlotterService:
    @staticmethod
    def plot_trace_solution(solution: SolutionResult, problem: FractionalProblem) -> None:
        x_values, u_values = TraceSolution.extract(solution.vertices, solution.values)
        plt.figure()
        plt.plot(x_values, u_values, label="FEM solution")
        plt.plot(x_values, problem.exact_solution(x_values), "--", label="Exact solution")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title("Solution on y = 0")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_solution(solution: SolutionResult, title: str = "Solution") -> None:
        x_values = solution.vertices[:, 0]
        y_values = solution.vertices[:, 1]
        z_values = solution.values.reshape(-1)
        fig = plt.figure(figsize=(10, 6))
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        surface = axis.plot_trisurf(
            x_values,
            y_values,
            z_values,
            triangles=solution.triangles,
            cmap="viridis",
        )
        fig.colorbar(surface, ax=axis, shrink=0.5, aspect=12)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("U")
        plt.tight_layout()
        plt.show()
