import numpy as np
import triangle as tr

class TriangulationService: 
    @staticmethod
    def triangulate_area(boundary_points: np.ndarray, triangulation_options: str):
        vertices_np = np.array(boundary_points)
        
        A = dict(vertices=vertices_np)
        B = tr.triangulate(A, triangulation_options)

        vertices = B["vertices"]
        triangles = B["triangles"]
        triangle_vertices = np.array([[vertices[j] for j in i] for i in triangles])
        
        return vertices, triangles, triangle_vertices, B

