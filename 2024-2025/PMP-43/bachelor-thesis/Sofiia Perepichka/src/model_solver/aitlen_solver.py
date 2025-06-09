import numpy as np
from src.data.simulation_data import SimulationData
from src.fem.aitken_order import AitkenOrder
from src.model_solver.solver import SimulationSolver
from src.model_solver import *
from src.plotter.plotter_service import PlotterService
from src.triangulation.triangulation import TriangulationService

class AitkenSolver: 
    @staticmethod
    def solve_aitken(boundary_points: np.ndarray, simulationData: SimulationData, start_area = 0.1, N_result = 2):
        levels = N_result + 2
        max_areas = [start_area / (4 ** i) for i in range(levels)]
        results = []

        for area in max_areas:
            print(f"\nMax_area = {area:.6f}")
            vertices, _, _, tri_data = \
                TriangulationService.triangulate_area(boundary_points, f"qa{area}")
            PlotterService.plot_mesh(tri_data)

            C, S, M, K_C, K_S = SimulationSolver.solve_time_dependent(tri_data, simulationData)

            results.append((vertices, C, M, K_C, S, K_S))

        print("\nOrder of convergence (Eitken) for component C")
        for i in range(len(results) - 2):
            v1, u1_C, _, _, _, _      = results[i]
            v2, u2_C, M2, K2_C, _, _      = results[i+1]
            v3, u3_C, M3, K3_C, _, _      = results[i+2]

            p_L2_C, p_H1_C = AitkenOrder.aitken_order_interpolated(
                v1, u1_C,
                v2, u2_C, M2, K2_C,
                v3, u3_C, M3, K3_C
            )

            if p_L2_C is None:
                print(f"Impossible")
            else:
                print(f"p_L2 = {p_L2_C:.4f}, p_H1 = {p_H1_C:.4f}")

        print("\nOrder of convergence (Eitken) for component S")
        for i in range(len(results) - 2):
            v1, _, _, _, u1_S, _ = results[i]
            v2, _, M2, _, u2_S, K2_S = results[i+1]
            v3, _, M3, _, u3_S, K3_S = results[i+2]

            p_L2_S, _ = AitkenOrder.aitken_order_interpolated(
                v1, u1_S,
                v2, u2_S, M2, K2_S,
                v3, u3_S, M3, K3_S
            )

            if p_L2_S is None:
                print(f"Impossible")
            else:
                print(f"p_L2 = {p_L2_C:.4f}, p_H1 = {p_H1_C:.4f}")