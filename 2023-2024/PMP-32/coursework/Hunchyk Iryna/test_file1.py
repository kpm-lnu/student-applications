import unittest
import numpy as np
from file1 import (
    compute_ke,
    compute_area,
    compute_me,
    compute_qe,
    assemble_global_matrix,
    assemble_rhs,
    exact,
    L2_Error
)


class TestFile1(unittest.TestCase):

    def setUp(self):
        self.triangle_vertices = np.array([[0, 0], [1, 0], [0, 1]])
        self.a11 = 1
        self.a22 = 1
        self.fe = [3, 3, 3]

    def test_compute_area(self):
        area = compute_area(self.triangle_vertices)
        self.assertAlmostEqual(area, 1.0, places=5)

    def test_compute_ke(self):
        ke = compute_ke(self.triangle_vertices, self.a11, self.a22)
        self.assertEqual(ke.shape, (3, 3))

    def test_compute_me(self):
        me = compute_me(self.triangle_vertices)
        self.assertEqual(me.shape, (3, 3))

    def test_compute_qe(self):
        qe = compute_qe(self.triangle_vertices, self.fe)
        self.assertEqual(len(qe), 3)

    def test_assemble_global_matrix(self):
        global_matrix = np.zeros((3, 3))
        stiffness_matrix = np.eye(3)
        triangle = [0, 1, 2]
        assembled_matrix = assemble_global_matrix(global_matrix, stiffness_matrix, triangle)
        self.assertTrue((assembled_matrix == stiffness_matrix).all())

    def test_assemble_rhs(self):
        global_b = np.zeros(3)
        qe = np.array([1, 2, 3])
        triangle = [0, 1, 2]
        assembled_rhs = assemble_rhs(qe, triangle, global_b)
        self.assertTrue((assembled_rhs == qe).all())

    def test_exact(self):
        x = 1.5
        expected_result = exact(x)
        self.assertAlmostEqual(expected_result, 2.25, places=5)

    def test_L2_Error(self):
        solution = np.array([1, 1, 1])
        tSolution = np.array([1, 1, 1])
        error = L2_Error(solution, tSolution)
        self.assertAlmostEqual(error, 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
