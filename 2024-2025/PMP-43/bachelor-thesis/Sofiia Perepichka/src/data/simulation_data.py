from typing import Callable
import numpy as np


class SimulationData:
    def __init__(self, triangulation_options: str, boundary_points: np.ndarray) -> None:
        self.triangulation_options = triangulation_options
        self.boundary_points = boundary_points
        self.stop_time = 1
        self.time_step = 0.1
       
        
        self.diffusion_cell_coefficient = 1.5
        self.diffusion_nutrient_coefficient = 0.1
        self.proliferation_coefficient = 1.7
        self.necrosis_coefficient = 0.5
        self.nutrient_consumption_coefficient = 0
        self.boundary_transfer_coefficient = 10000000
        self.nutrient_concentration = 1
        self.nutrient_source = 1
        
        self.initial_cell_density: Callable[[float, float], float] = lambda x, y: 1
        self.initial_nutrient_concentration: Callable[[float, float], float] = lambda x, y: 1