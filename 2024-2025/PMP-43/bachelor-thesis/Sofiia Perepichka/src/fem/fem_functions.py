import numpy as np


from src.common.helpers import HelperFunctions

class FemService: 
    
    @staticmethod
    def compute_stiffness_matrix(triangle_vertices: np.ndarray, a11: float, a22: float) -> np.ndarray:
      (x1, y1), (x2, y2), (x3, y3) = triangle_vertices

      x_diff_12 = x1 - x2
      x_diff_23 = x2 - x3
      x_diff_31 = x3 - x1
      y_diff_12 = y1 - y2
      y_diff_23 = y2 - y3
      y_diff_31 = y3 - y1
      
      delta = HelperFunctions.compute_delta(triangle_vertices)
      
      stiffness_matrix = (1 / (2 * delta)) * np.array([
        [a11 * y_diff_23**2 + a22 * x_diff_23**2,
         a11 * y_diff_23 * y_diff_31 + a22 * x_diff_23 * x_diff_31,
         a11 * y_diff_23 * y_diff_12 + a22 * x_diff_23 * x_diff_12],

        [a11 * y_diff_31 * y_diff_23 + a22 * x_diff_31 * x_diff_23,
         a11 * y_diff_31**2 + a22 * x_diff_31**2,
         a11 * y_diff_31 * y_diff_12 + a22 * x_diff_31 * x_diff_12],

        [a11 * y_diff_12 * y_diff_23 + a22 * x_diff_12 * x_diff_23,
         a11 * y_diff_12 * y_diff_31 + a22 * x_diff_12 * x_diff_31,
         a11 * y_diff_12**2 + a22 * x_diff_12**2]])
      
      
      return stiffness_matrix


    @staticmethod
    def compute_mass_matrix(triangle_vertices: np.ndarray) -> np.ndarray:
        M = np.array([[2, 1, 1], 
                      [1, 2, 1], 
                      [1, 1, 2]])
        delta = HelperFunctions.compute_delta(triangle_vertices)
        mass_matrix = (delta / 24) * M
        return mass_matrix
    
    @staticmethod
    def compute_boundary_matrix(boundary_length: float, beta: float, beta_coef: float) -> np.ndarray:
        matrix = np.array([
            [2, 1],
            [1, 2]])
        
        if beta == 0:
           return np.zeros_like(matrix)
       
        boundary_matrix_matrix = (beta_coef  / beta) * (boundary_length / 6) * matrix
        return boundary_matrix_matrix

    @staticmethod
    def compute_boundary_vector(boundary_length: float, psi: float, beta: float, beta_coef: float) -> np.ndarray:
       vector = np.array([[1],[1]])
       
       if beta == 0:
        return np.zeros_like(vector)
    
       boundary_vector = beta_coef * (psi / beta) * (boundary_length / 2) * vector
       return boundary_vector
   
    @staticmethod
    def compute_source_vector(triangle_vertices: np.ndarray, fe: list) -> np.ndarray:
        mass_matrix = FemService.compute_mass_matrix(triangle_vertices)
        source_vector = np.dot(mass_matrix,np.array(fe).T)
        return source_vector
    
    @staticmethod
    def compute_global_matrix(global_matrix: np.ndarray, matrix: np.ndarray, triangle: np.ndarray) -> np.ndarray:
        for i in range(3):
            for j in range(3):
                global_matrix[triangle[i], triangle[j]] += matrix[i, j] 
        
        return global_matrix
    
    @staticmethod
    def compute_global_matrix_boundary(global_matrix: np.ndarray, matrix: np.ndarray, edge: np.ndarray) -> np.ndarray:
        for i in range(2):
            for j in range(2):
                global_matrix[edge[i], edge[j]] += matrix[i, j] 
        return global_matrix
    
    @staticmethod
    def compute_global_vector(Qe: np.ndarray, triangle: np.ndarray, global_vector: np.ndarray) -> np.ndarray:
        for i in range(3):
            global_vector[triangle[i]] += Qe[i]
        return global_vector
    
    @staticmethod
    def compute_global_vector_boundary(Qe: np.ndarray, edge: np.ndarray, globalB: np.ndarray) -> np.ndarray:
        for i in range(2):
            globalB[edge[i]] += Qe[i] 
        return globalB
