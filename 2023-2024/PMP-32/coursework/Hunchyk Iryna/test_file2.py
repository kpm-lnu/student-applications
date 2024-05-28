import unittest
import numpy as np
import triangle as tr
from file1 import (
    compute_ke,
    compute_qe,
    assemble_global_matrix,
    assemble_rhs,
    exact,
    L2_Error
)
from plotter import plot_mesh


class TestFile2(unittest.TestCase):

    def setUp(self):
        self.rectangle_vertices = [[0, 0], [3, 0], [0, 1], [3, 1]]
        A = dict(vertices=np.array(self.rectangle_vertices), segments=tr.convex_hull(self.rectangle_vertices))
        self.B = tr.triangulate(A, "Da0.08 q30p")
        self.vertices = self.B["vertices"]
        self.triangles = self.B["triangles"]
        self.triangle_vertices = np.array([[self.vertices[j] for j in i] for i in self.triangles])

    def test_assembled_system(self):
        assembled_system = np.zeros((len(self.vertices), len(self.vertices)))
        rhs = np.zeros(len(self.vertices))
        for i in range(len(self.triangles)):
            ke = np.array(compute_ke(self.triangle_vertices[i], a11=1, a22=1))
            qe = compute_qe(self.triangle_vertices[i], fe=[3, 3, 3])
            assembled_system = assemble_global_matrix(assembled_system, ke, self.triangles[i])
            rhs = assemble_rhs(qe, self.triangles[i], rhs)
        for i in range(len(self.vertices)):
            if self.vertices[i, 0] == 0 or self.vertices[i, 0] == 3:
                assembled_system[i, i] = 1000000000000
        solution = np.linalg.solve(assembled_system, rhs)
        u_exact = [exact(self.vertices[i, 0]) for i in range(len(self.vertices))]
        error_L2 = L2_Error(solution, u_exact)
        self.assertLess(error_L2, 1e-5)


if __name__ == '__main__':
    unittest.main()
