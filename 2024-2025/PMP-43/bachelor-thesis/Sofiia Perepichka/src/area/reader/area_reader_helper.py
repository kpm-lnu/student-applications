import numpy as np

class AreaReaderHelper:
    @staticmethod
    def read_boundary_points_from_file(file_path: str)-> np.ndarray :
        boundary_points = np.loadtxt(file_path, usecols=(0, 1))
        return boundary_points
