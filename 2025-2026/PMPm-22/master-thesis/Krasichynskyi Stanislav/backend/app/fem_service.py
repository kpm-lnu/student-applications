import numpy as np

from .compute_fucns import (
    assemble_global_matrix,
    assemble_rhs,
    compute_ke,
    compute_pressure,
    compute_qe,
)


def solve_concentration(area, a_11=1.0, a_22=1.0, force_value=0.1, boundary_value=1.0):
    if area.tri is None:
        raise ValueError("Спочатку треба виконати триангуляцію")

    vertices = area.tri["vertices"]
    triangles = area.tri["triangles"]

    triangle_vertices = np.array([[vertices[j] for j in tri] for tri in triangles])

    assembled_system = np.zeros((len(vertices), len(vertices)))
    rhs = np.zeros(len(vertices))

    all_edges = set(map(tuple, map(sorted, area.segments)))
    boundary_points = {pt for edge in all_edges for pt in edge}

    for i in range(len(triangles)):
        ke = np.array(
            compute_ke(
                triangle_vertices[i],
                a_11=a_11,
                a_22=a_22,
            )
        )
        qe = compute_qe(triangle_vertices[i], fe=[force_value, force_value, force_value])

        assembled_system = assemble_global_matrix(
            assembled_system, ke, triangles[i]
        )
        rhs = assemble_rhs(qe, triangles[i], rhs)

    boundary_penalty = 1e7 * boundary_value if boundary_value != 0 else 1e7

    for i in range(len(vertices)):
        if i in boundary_points:
            assembled_system[i, :] = 0.0
            assembled_system[i, i] = 1e7
            rhs[i] = boundary_penalty

    solution = np.linalg.solve(assembled_system, rhs)
    return vertices, triangles, solution


def solve_pressure(vertices, concentration, A=1.0, G=1.0, chi=0.0, k=0.0, d=2):
    pressure = np.array(
        [
            compute_pressure(A=A, G=G, c=float(c), chi=chi, x=v, d=d, k=k)
            for v, c in zip(vertices, concentration)
        ]
    )
    return pressure


def solve_fields(
    area,
    a_11=1.0,
    a_22=1.0,
    force_value=0.1,
    boundary_value=1.0,
    pressure_a=1.0,
    pressure_g=1.0,
    pressure_chi=0.0,
    pressure_k=0.0,
    pressure_dimension=2,
):
    vertices, triangles, concentration = solve_concentration(
        area,
        a_11=a_11,
        a_22=a_22,
        force_value=force_value,
        boundary_value=boundary_value,
    )
    pressure = solve_pressure(
        vertices,
        concentration,
        A=pressure_a,
        G=pressure_g,
        chi=pressure_chi,
        k=pressure_k,
        d=pressure_dimension,
    )
    return vertices, triangles, concentration, pressure
