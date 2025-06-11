import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.mesh_generator import generate_mesh
from core.fem_solver import assemble_global_matrix, apply_dirichlet, solve_system
from core.exact_solution import exact_solution_vector
from core.error_metrics import l2_error
from core.exact_solution import rhs_function


def test_fem_solution_accuracy():
    nodes, elements, boundary_nodes = generate_mesh(5, 5)
    K, F = assemble_global_matrix(nodes, elements, f_func=rhs_function)
    K, F = apply_dirichlet(K, F, boundary_nodes)
    u_h = solve_system(K, F)
    u_exact = exact_solution_vector(nodes)
    error = l2_error(u_h, u_exact)
    assert error < 0.3, f"Завелика похибка FEM: {error}"
