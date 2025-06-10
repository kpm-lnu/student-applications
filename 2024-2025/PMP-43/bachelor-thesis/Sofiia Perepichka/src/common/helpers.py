import numpy as np

class HelperFunctions: 
    
    @staticmethod
    def compute_delta(triangle_vertices: np.ndarray) -> float:
        x1, y1 = triangle_vertices[0]
        x2, y2 = triangle_vertices[1]
        x3, y3 = triangle_vertices[2]
        
        area = 0.5 * ((x1 * y2 + x2 * y3 + x3 * y1) - (y1 * x2 + y2 * x3 + y3 * x1))
        delta = 2 * area

        return delta
    
    @staticmethod
    def compute_boundary_length(vertex1: np.ndarray, vertex2: np.ndarray) -> float:
        boundary_length = np.linalg.norm(np.array(vertex1) - np.array(vertex2))
        return boundary_length

    @staticmethod
    def get_boundary_edges(triangles: np.ndarray) -> list:
        edge_counts = {}
        
        for tri in triangles:
            edges = [(tri[0], tri[1]),(tri[1], tri[2]), (tri[2], tri[0])]
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                if sorted_edge in edge_counts:
                    edge_counts[sorted_edge] += 1
                else:
                    edge_counts[sorted_edge] = 1
                    
        boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
        return boundary_edges
