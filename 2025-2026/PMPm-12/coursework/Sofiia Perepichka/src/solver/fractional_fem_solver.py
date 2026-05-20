from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from src.common.helpers import HelperFunctions
from src.fem.fem_functions import FemService
from src.problem.fractional_problem import FractionalProblem
from src.solver.solution_result import SolutionResult
from src.triangulation.triangulation import TriangulationService


@dataclass(slots=True)
class FractionalFemSolver:
    problem: FractionalProblem

    def solve(self, n_x: int, n_y: int | None = None) -> SolutionResult:
        n_y = n_x if n_y is None else n_y
        config = self.problem.config
        mesh = TriangulationService.generate_fractional_mesh(
            L=config.L,
            Y=config.Y,
            N=n_x,
            M=n_y,
            gamma=config.gamma_mesh,
        )
        vertices = mesh["vertices"]
        triangles = mesh["triangles"]
        matrix, rhs = self._assemble_system(vertices, triangles)
        self._apply_dirichlet_boundary_conditions(matrix, rhs, vertices)
        solution = spsolve(matrix.tocsr(), rhs)

        if np.any(~np.isfinite(solution)):
            raise RuntimeError("NaN or Inf in numerical solution")

        return SolutionResult(
            mesh=mesh,
            vertices=vertices,
            triangles=triangles,
            values=solution,
        )

    def _assemble_system(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
    ) -> tuple[lil_matrix, np.ndarray]:
        node_count = len(vertices)
        matrix = lil_matrix((node_count, node_count), dtype=float)
        rhs = np.zeros(node_count, dtype=float)
        self._assemble_stiffness_matrix(matrix, vertices, triangles)
        self._assemble_bottom_source_vector(rhs, vertices, triangles)
        return matrix, rhs

    def _assemble_stiffness_matrix(
        self,
        matrix: lil_matrix,
        vertices: np.ndarray,
        triangles: np.ndarray,
    ) -> None:
        for triangle in triangles:
            triangle_vertices = vertices[triangle]
            local_matrix = FemService.compute_stiffness_matrix(
                triangle_vertices=triangle_vertices,
                a11=1.0,
                a22=1.0,
                alpha=self.problem.config.alpha,
            )
            FemService.compute_global_matrix(matrix, local_matrix, triangle)

    def _assemble_bottom_source_vector(
        self,
        rhs: np.ndarray,
        vertices: np.ndarray,
        triangles: np.ndarray,
    ) -> None:
        boundary_edges = HelperFunctions.get_boundary_edges(triangles)

        for edge in boundary_edges:
            p1 = vertices[edge[0]]
            p2 = vertices[edge[1]]

            if self._is_bottom_edge(p1, p2):
                edge_vertices = np.array([p1, p2])
                local_vector = FemService.compute_source_vector(
                    edge_vertices=edge_vertices,
                    source_function=self.problem.source,
                    ds=self.problem.config.d_s,
                )
                FemService.compute_global_vector(local_vector.flatten(), edge, rhs)

    def _apply_dirichlet_boundary_conditions(
        self,
        matrix: lil_matrix,
        rhs: np.ndarray,
        vertices: np.ndarray,
    ) -> None:
        config = self.problem.config

        for index, (x, y) in enumerate(vertices):
            is_left = abs(x) < 1e-12
            is_right = abs(x - config.L) < 1e-12
            is_top = abs(y - config.Y) < 1e-12

            if is_left or is_right or is_top:
                matrix.rows[index] = [index]
                matrix.data[index] = [1.0]
                rhs[index] = 0.0

    @staticmethod
    def _is_bottom_edge(p1: np.ndarray, p2: np.ndarray) -> bool:
        return abs(p1[1]) < 1e-12 and abs(p2[1]) < 1e-12
