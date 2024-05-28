import unittest
import numpy as np
from plotter import plot_mesh
import triangle as tr


class TestPlotter(unittest.TestCase):

    def test_plot_mesh(self):
        rectangle_vertices = [[0, 0], [3, 0], [0, 1], [3, 1]]
        A = dict(vertices=np.array(rectangle_vertices), segments=tr.convex_hull(rectangle_vertices))
        B = tr.triangulate(A, "Da0.08 q30p")
        # Just check if the function runs without errors
        try:
            plot_mesh(B)
            success = True
        except Exception as e:
            success = False
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
