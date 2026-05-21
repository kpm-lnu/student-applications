import numpy as np
from pathlib import Path

from fem_elasticity.material import elasticity_matrix
from fem_elasticity.mesh import quarter_plate_with_hole_mesh, nodes_on_x, nodes_on_y, classify_edges_by_midpoint
from fem_elasticity.assembly import global_force_vector, add_edge_traction
from fem_elasticity.boundary_conditions import fixed_dofs_for_nodes
from fem_elasticity.solver import solve_elasticity
from fem_elasticity.postprocessing import (
    ensure_dir, save_text_report, plot_mesh, plot_deformed_mesh,
    plot_nodal_field, plot_von_mises, displacement_components
)


def run(output_dir: str | Path = "results/plate_with_hole", n_rect: int = 30) -> dict:
    out = ensure_dir(output_dir)

    L = 1.0
    H = 1.0
    R = 0.12
    thickness = 0.01
    E = 2.1e11
    nu = 0.30
    p = 1.0e6
    plane_type = "stress"

    mesh = quarter_plate_with_hole_mesh(L, H, R, n_rect=n_rect, n_theta=44, n_radial=12)
    D = elasticity_matrix(E, nu, plane_type)

    F = global_force_vector(len(mesh.nodes))

    ymax = H / 2.0
    top_edges = classify_edges_by_midpoint(mesh, lambda q: np.isclose(q[1], ymax, atol=1e-8))
    for edge in top_edges:
        add_edge_traction(F, mesh.nodes, edge, np.array([0.0, p]), thickness)

    x0_nodes = nodes_on_x(mesh, 0.0, tol=1e-10)
    y0_nodes = nodes_on_y(mesh, 0.0, tol=1e-10)
    fixed = np.unique(np.concatenate([
        fixed_dofs_for_nodes(x0_nodes, (True, False)),
        fixed_dofs_for_nodes(y0_nodes, (False, True)),
    ]))

    result = solve_elasticity(mesh, D, thickness, F, fixed)

    params = {
        "Розмір пластини L, м": L,
        "Розмір пластини H, м": H,
        "Радіус отвору R, м": R,
        "Товщина t, м": thickness,
        "Модуль Юнга E, Па": f"{E:.3e}",
        "Коефіцієнт Пуассона nu": nu,
        "Розтягувальне навантаження p, Па": f"{p:.3e}",
        "Тип плоскої задачі": "плоский напружений стан",
        "n_rect": n_rect,
    }

    save_text_report(out / "zvit.txt", "Пластина з круглим отвором під розтягом", params, result)

    ux, uy, mag = displacement_components(result.displacements)
    plot_mesh(mesh, out / "sitka.png", "Скінченно-елементна сітка пластини з круглим отвором")
    plot_deformed_mesh(mesh, result.displacements, out / "deformovana_sitka.png", title="Деформована сітка пластини з отвором")
    plot_nodal_field(mesh, mag, out / "pole_peremishchen.png", "Поле модуля переміщень", "Модуль переміщення, м")
    plot_von_mises(mesh, result, out / "napruzhennia_mizesa.png", "Поле напружень за Мізесом для пластини з отвором")

    nominal_stress = p
    kt_numeric = result.max_von_mises / nominal_stress

    print("\n" + "=" * 72)
    print("ПРИКЛАД 1: ПЛАСТИНА З КРУГЛИМ ОТВОРОМ ПІД РОЗТЯГОМ")
    print("=" * 72)
    print(f"Кількість вузлів: {result.n_nodes}")
    print(f"Кількість елементів: {result.n_elements}")
    print(f"Максимальне переміщення: {result.max_displacement:.6e} м")
    print(f"Максимальне напруження за Мізесом: {result.max_von_mises:.6e} Па")
    print(f"Оціночний коефіцієнт концентрації напружень: {kt_numeric:.3f}")
    print(f"Час розрахунку: {result.elapsed_time:.6f} с")
    print(f"Результати збережено в папці: {out}")

    return {
        "Назва задачі": "Пластина з отвором",
        "Вузли": result.n_nodes,
        "Елементи": result.n_elements,
        "Максимальне переміщення, м": f"{result.max_displacement:.6e}",
        "Максимальне напруження за Мізесом, Па": f"{result.max_von_mises:.6e}",
        "Час, с": f"{result.elapsed_time:.6f}",
        "Додатково": f"K_t ≈ {kt_numeric:.3f}",
    }


if __name__ == "__main__":
    run()
