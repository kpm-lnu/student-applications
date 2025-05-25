import numpy as np


class MeshData:
    def __init__(
        self, vertices=np.array([]), vertex_markers=np.array([]), triangles=np.array([])
    ):
        self.vertices = np.array(vertices)
        self.vertex_markers = np.array(vertex_markers)
        self.triangles = np.array(triangles)

    def add_vertex(self, vertex, marker):
        self.vertices = np.append(self.vertices, [vertex], axis=0)
        self.vertex_markers = np.append(self.vertex_markers, [[marker]], axis=0)

    def add_triangle(self, triangle):
        self.triangles = np.append(self.triangles, [triangle], axis=0)

    def vertex_count(self):
        return self.vertices.shape[0]

    def triangle_count(self):
        return self.triangles.shape[0]

    def __str__(self):
        return f"MeshData with {self.vertex_count()} vertices and {self.triangle_count()} triangles"
