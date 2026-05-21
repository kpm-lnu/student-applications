import numpy as np
from pathlib import Path

from fem_elasticity.material import elasticity_matrix
from fem_elasticity.mesh import rectangular_tri_mesh, nodes_on_x, classify_edges_by_midpoint
from fem_elasticity.assembly import global_force_vector, add_edge_traction
from fem_elasticity.boundary_conditions import fixed_dofs_for_nodes
from fem_elasticity.solver import solve_elasticity
from fem_elasticity.postprocessing import (
    ensure_dir, save_text_report, plot_mesh, plot_deformed_mesh,
    plot_nodal_field, plot_von_mises, displacement_components
)


def run(output_dir: str | Path = "results/cantilever_beam", nx: int = 42, ny: int = 10) -> dict:
    out = ensure_dir(output_dir)

    L = 1.0
    H = 0.20
    thickness = 0.01
    E = 2.1e11
    nu = 0.30
    P = 1000.0
    plane_type = "stress"

    mesh = rectangular_tri_mesh(L, H, nx, ny)
    D = elasticity_matrix(E, nu, plane_type)

    F = global_force_vector(len(mesh.nodes))
    traction_y = -P / (thickness * H)
    right_edges = classify_edges_by_midpoint(mesh, lambda p: np.isclose(p[0], L, atol=1e-9))
    for edge in right_edges:
        add_edge_traction(F, mesh.nodes, edge, np.array([0.0, traction_y]), thickness)

    left_nodes = nodes_on_x(mesh, 0.0)
    fixed = fixed_dofs_for_nodes(left_nodes, (True, True))

    result = solve_elasticity(mesh, D, thickness, F, fixed)

    params = {
        "Довжина L, м": L,
        "Висота H, м": H,
        "Товщина t, м": thickness,
        "Модуль Юнга E, Па": f"{E:.3e}",
        "Коефіцієнт Пуассона nu": nu,
        "Повна вертикальна сила P, Н": P,
        "Тип плоскої задачі": "плоский напружений стан",
        "nx": nx,
        "ny": ny,
    }

    save_text_report(out / "zvit.txt", "Консольна балка під дією вертикальної сили", params, result)

    ux, uy, mag = displacement_components(result.displacements)
    plot_mesh(mesh, out / "sitka.png", "Скінченно-елементна сітка консольної балки")
    plot_deformed_mesh(mesh, result.displacements, out / "deformovana_sitka.png", title="Деформована сітка консольної балки")
    plot_nodal_field(mesh, mag, out / "pole_peremishchen.png", "Поле модуля переміщень", "Модуль переміщення, м")
    plot_nodal_field(mesh, uy, out / "verticalni_peremishchennia.png", "Поле вертикальних переміщень", "Вертикальне переміщення, м")
    plot_von_mises(mesh, result, out / "napruzhennia_mizesa.png", "Поле напружень за Мізесом для консольної балки")

    I = thickness * H ** 3 / 12.0
    w_an = P * L ** 3 / (3.0 * E * I)
    w_fem = abs(float(np.min(uy)))
    error = abs(w_fem - w_an) / abs(w_an) * 100.0

    print("\n" + "=" * 72)
    print("ПРИКЛАД 2: КОНСОЛЬНА БАЛКА ПІД ДІЄЮ ВЕРТИКАЛЬНОЇ СИЛИ")
    print("=" * 72)
    print(f"Кількість вузлів: {result.n_nodes}")
    print(f"Кількість елементів: {result.n_elements}")
    print(f"Максимальне переміщення: {result.max_displacement:.6e} м")
    print(f"Максимальне напруження за Мізесом: {result.max_von_mises:.6e} Па")
    print(f"Прогин за МСЕ: {w_fem:.6e} м")
    print(f"Аналітичний прогин: {w_an:.6e} м")
    print(f"Відносна похибка прогину: {error:.3f} %")
    print(f"Час розрахунку: {result.elapsed_time:.6f} с")
    print(f"Результати збережено в папці: {out}")

    return {
        "Назва задачі": "Консольна балка",
        "Вузли": result.n_nodes,
        "Елементи": result.n_elements,
        "Максимальне переміщення, м": f"{result.max_displacement:.6e}",
        "Максимальне напруження за Мізесом, Па": f"{result.max_von_mises:.6e}",
        "Час, с": f"{result.elapsed_time:.6f}",
        "Додатково": f"Похибка прогину: {error:.3f} %",
    }


if __name__ == "__main__":
    run()
