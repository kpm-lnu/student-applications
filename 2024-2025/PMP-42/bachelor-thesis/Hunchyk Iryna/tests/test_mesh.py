import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.mesh_generator import generate_mesh


def test_mesh_node_count():
    nx, ny = 5, 5
    nodes, elements, boundary_nodes = generate_mesh(nx, ny)
    assert len(nodes) == (nx + 1) * (ny + 1), "Некоректна кількість вузлів"
    assert len(elements) == 2 * nx * ny, "Некоректна кількість елементів"
    assert len(boundary_nodes) > 0, "Не виявлено граничних вузлів"
