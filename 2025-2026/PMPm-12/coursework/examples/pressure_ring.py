import numpy as np
from pathlib import Path

from fem_elasticity.material import elasticity_matrix
from fem_elasticity.mesh import quarter_ring_mesh, nodes_on_x, nodes_on_y, classify_edges_by_midpoint
from fem_elasticity.assembly import global_force_vector, add_edge_traction_function
from fem_elasticity.boundary_conditions import fixed_dofs_for_nodes
from fem_elasticity.solver import solve_elasticity
from fem_elasticity.postprocessing import (
    ensure_dir, save_text_report, plot_mesh, plot_deformed_mesh,
    plot_nodal_field, plot_von_mises, displacement_components
)


def run(output_dir: str | Path = "results/pressure_ring", nr: int = 16, nt: int = 48) -> dict:
    out = ensure_dir(output_dir)

    ri = 0.20
    ro = 0.50
    thickness = 0.01
    E = 2.1e11
    nu = 0.30
    p_i = 5.0e6
    p_o = 0.0
    plane_type = "strain"

    mesh = quarter_ring_mesh(ri, ro, nr, nt)
    D = elasticity_matrix(E, nu, plane_type)

    F = global_force_vector(len(mesh.nodes))

    inner_edges = classify_edges_by_midpoint(mesh, lambda p: np.isclose(np.linalg.norm(p), ri, atol=(ro - ri) / nr * 0.55))

    def inner_pressure(mid):
        r = np.linalg.norm(mid)
        e_r = mid / r
        return p_i * e_r

    for edge in inner_edges:
        add_edge_traction_function(F, mesh.nodes, edge, inner_pressure, thickness)

    x0_nodes = nodes_on_x(mesh, 0.0, tol=1e-10)
    y0_nodes = nodes_on_y(mesh, 0.0, tol=1e-10)
    fixed = np.unique(np.concatenate([
        fixed_dofs_for_nodes(x0_nodes, (True, False)),
        fixed_dofs_for_nodes(y0_nodes, (False, True)),
    ]))

    result = solve_elasticity(mesh, D, thickness, F, fixed)

    params = {
        "Внутрішній радіус ri, м": ri,
        "Зовнішній радіус ro, м": ro,
        "Товщина t, м": thickness,
        "Модуль Юнга E, Па": f"{E:.3e}",
        "Коефіцієнт Пуассона nu": nu,
        "Внутрішній тиск, Па": f"{p_i:.3e}",
        "Зовнішній тиск, Па": f"{p_o:.3e}",
        "Тип плоскої задачі": "плоска деформація",
        "nr": nr,
        "nt": nt,
    }

    save_text_report(out / "zvit.txt", "Товстостінне кільце під внутрішнім тиском", params, result)

    ux, uy, mag = displacement_components(result.displacements)
    plot_mesh(mesh, out / "sitka.png", "Скінченно-елементна сітка товстостінного кільця")
    plot_deformed_mesh(mesh, result.displacements, out / "deformovana_sitka.png", title="Деформована сітка товстостінного кільця")
    plot_nodal_field(mesh, mag, out / "pole_peremishchen.png", "Поле модуля переміщень", "Модуль переміщення, м")
    plot_von_mises(mesh, result, out / "napruzhennia_mizesa.png", "Поле напружень за Мізесом для товстостінного кільця")

    A = (p_i * ri ** 2 - p_o * ro ** 2) / (ro ** 2 - ri ** 2)
    Bc = ((p_i - p_o) * ri ** 2 * ro ** 2) / (ro ** 2 - ri ** 2)
    radii = np.array([ri, 0.5 * (ri + ro), ro])
    with (out / "porivniannia_lame.csv").open("w", encoding="utf-8") as f:
        f.write("r, м;sigma_r за Ламе, Па;sigma_theta за Ламе, Па\n")
        for r in radii:
            sigma_r = A - Bc / r ** 2
            sigma_theta = A + Bc / r ** 2
            f.write(f"{r:.6e};{sigma_r:.6e};{sigma_theta:.6e}\n")

    print("\n" + "=" * 72)
    print("ПРИКЛАД 3: ТОВСТОСТІННЕ КІЛЬЦЕ ПІД ВНУТРІШНІМ ТИСКОМ")
    print("=" * 72)
    print(f"Кількість вузлів: {result.n_nodes}")
    print(f"Кількість елементів: {result.n_elements}")
    print(f"Максимальне переміщення: {result.max_displacement:.6e} м")
    print(f"Максимальне напруження за Мізесом: {result.max_von_mises:.6e} Па")
    print("Порівняння за формулами Ламе збережено у файлі porivniannia_lame.csv")
    print(f"Час розрахунку: {result.elapsed_time:.6f} с")
    print(f"Результати збережено в папці: {out}")

    return {
        "Назва задачі": "Товстостінне кільце",
        "Вузли": result.n_nodes,
        "Елементи": result.n_elements,
        "Максимальне переміщення, м": f"{result.max_displacement:.6e}",
        "Максимальне напруження за Мізесом, Па": f"{result.max_von_mises:.6e}",
        "Час, с": f"{result.elapsed_time:.6f}",
        "Додатково": "Є порівняння за формулами Ламе",
    }


if __name__ == "__main__":
    run()
