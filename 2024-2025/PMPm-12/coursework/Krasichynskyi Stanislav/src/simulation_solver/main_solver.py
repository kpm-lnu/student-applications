from matplotlib import pyplot as plt
import numpy as np
from gui.components.window.final_result_window import ResultsWindow
from simulation_solver.math_helper import (
    assemble_global_matrix,
    assemble_rhs,
    compute_gradient,
    compute_ke,
    compute_qe,
    compute_pressure,
)


def start_simulation(triangulation_results, segments, math_model):
    vertices = triangulation_results["vertices"]
    triangles = triangulation_results["triangles"]

    edges = set()
    for tri in triangles:
        edges.add(tuple(sorted((tri[0], tri[1]))))
        edges.add(tuple(sorted((tri[1], tri[2]))))
        edges.add(tuple(sorted((tri[0], tri[2]))))

    all_edges = set(map(tuple, map(sorted, segments)))

    boundary_points = {pt for edge in all_edges for pt in edge}

    triangle_vertices = np.array([[vertices[j] for j in i] for i in triangles])

    assembled_system = np.zeros((len(vertices), len(vertices)))
    rhs = np.zeros(len(vertices))

    for i in range(len(triangles)):
        ke = np.array(
            compute_ke(
                triangle_vertices[i],
                a_11=math_model.diffusion_coefficient,
                a_22=math_model.diffusion_coefficient,
            )
        )

        ## Question about math model
        qe = compute_qe(triangle_vertices[i], fe=[0.1, 0.1, 0.1])

        assembled_system = assemble_global_matrix(assembled_system, ke, triangles[i])
        rhs = assemble_rhs(qe, triangles[i], rhs)

    for i in range(len(vertices)):
        if i in boundary_points or i in boundary_points:
            assembled_system[i, :] = 0
            assembled_system[i, i] = 1e7
            rhs[i] = 1e7

    concentration_solution = np.linalg.solve(assembled_system, rhs)

    X_concentration = triangulation_results["vertices"][:, 0]
    Y_concentration = triangulation_results["vertices"][:, 1]
    Z_concentration = concentration_solution

    ## pressure
    vertices = triangulation_results["vertices"]
    triangles = triangulation_results["triangles"]

    edges = set()
    for tri in triangles:
        edges.add(tuple(sorted((tri[0], tri[1]))))
        edges.add(tuple(sorted((tri[1], tri[2]))))
        edges.add(tuple(sorted((tri[0], tri[2]))))

    all_edges = set(map(tuple, map(sorted, segments)))

    boundary_points = {pt for edge in all_edges for pt in edge}

    triangle_vertices = np.array([[vertices[j] for j in i] for i in triangles])

    assembled_system = np.zeros((len(vertices), len(vertices)))
    rhs = np.zeros(len(vertices))

    for i in range(len(triangles)):
        ke = np.array(
            compute_ke(
                triangle_vertices[i],
                a_11=1,
                a_22=1,
            )
        )

        qe = compute_qe(triangle_vertices[i], fe=[1, 1, 1])

        assembled_system = assemble_global_matrix(assembled_system, ke, triangles[i])
        rhs = assemble_rhs(qe, triangles[i], rhs)

    for i in range(len(vertices)):
        if i in boundary_points or i in boundary_points:
            x = vertices[i, 0]
            y = vertices[i, 1]

            pressure_value = compute_pressure(
                math_model.adhesion_measure,
                math_model.apoptosis_measure,
                Z_concentration[i],
                -0.65,
                [x, y],
            )
            assembled_system[i, :] = 0
            assembled_system[i, i] = 1e7
            rhs[i] = 1e7 * pressure_value

    pressure_solution = np.linalg.solve(assembled_system, rhs)

    X_concentration = triangulation_results["vertices"][:, 0]
    Y_concentration = triangulation_results["vertices"][:, 1]
    Z_pressure = pressure_solution

    boundary_indices = {pt for edge in all_edges for pt in edge}   
    concentration_data = (X_concentration, Y_concentration, Z_concentration)
    pressure_data = (X_concentration, Y_concentration, Z_pressure)

    results_window = ResultsWindow(boundary_indices, vertices, Z_pressure, Z_concentration)
    results_window.mainloop()

    return X_concentration, Y_concentration, Z_concentration
