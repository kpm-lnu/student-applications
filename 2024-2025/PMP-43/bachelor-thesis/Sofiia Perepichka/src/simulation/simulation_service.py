from src.model_solver.solver import SimulationSolver
from src.triangulation.triangulation import TriangulationService
from src.plotter.plotter_service import PlotterService
from src.data.simulation_data import SimulationData


class SimulationService:
    @staticmethod
    def start_simulation(simulation_data: SimulationData) -> None:
        
        _, _, _ , full_triangulation_data = TriangulationService.triangulate_area(simulation_data.boundary_points, simulation_data.triangulation_options)

        PlotterService.plot_mesh(full_triangulation_data)
        
        SimulationSolver.solve_time_dependent(full_triangulation_data, simulation_data)


    