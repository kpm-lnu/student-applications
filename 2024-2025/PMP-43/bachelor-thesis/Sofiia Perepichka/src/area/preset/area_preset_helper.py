import numpy as np

class AreaPresetHelper:
    @staticmethod
    def generate_circle_boundary(radius: float = 1, num_points: int = 120) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        boundary_points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        
        return boundary_points