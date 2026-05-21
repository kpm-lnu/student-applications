import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from src.assembly import assemble_all
from src.mesh import TriangularMesh
from src.solver import Solver
from src.stabilization import assemble_supg_mass_matrix, assemble_supg_matrix
from src.utils import compute_peclet_number, l2_error

OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)


def example_1_diffusion():
    print("\n" + "=" * 50)
    print("  Приклад 1. Чиста дифузія")
    print("=" * 50)

    D = 0.01
    T = 0.5
    dt = 0.005
    nx = ny = 32

    u0_func = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    u_exact = lambda x, y, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2 * np.pi**2 * D * t)
    g_func = lambda x, y, t: 0.0

    mesh = TriangularMesh(1.0, 1.0, nx, ny)
    M, K, C, _ = assemble_all(mesh, D, [0.0, 0.0])
    u0 = np.array([u0_func(x, y) for x, y in mesh.nodes])

    solver = Solver(mesh, M, K, C, dt, theta=1.0)
    solutions, times = solver.solve(u0, T, g_func=g_func, store_every=25)

    u_ex_final = np.array([u_exact(x, y, times[-1]) for x, y in mesh.nodes])

    err = l2_error(mesh, solutions[-1], u_exact, times[-1])
    print(f"  Сітка: {nx}x{ny}, вузлів: {mesh.n_nodes}")
    print(f"  L2-похибка при t={times[-1]:.2f}: {err:.4e}")

    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    snap_idx = [0, len(solutions) // 2, -1]
    titles = [f't = {times[i]:.2f}' for i in snap_idx]
    titles.append(f'Точний, t = {times[-1]:.2f}')

    data = [solutions[i] for i in snap_idx] + [u_ex_final]
    vmin, vmax = 0, 1

    for ax, u, title in zip(axes, data, titles):
        tcf = ax.tricontourf(triang, u, levels=np.linspace(vmin, vmax, 15), cmap='viridis')
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.colorbar(tcf, ax=axes, shrink=0.8, label='u')
    fig.suptitle('Приклад 1. Чиста дифузія (D = 0.01)', fontsize=13, y=1.02)
    fig.savefig(os.path.join(OUTDIR, "example_1_diffusion.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {OUTDIR}/example_1_diffusion.png")


def example_2_transport():
    print("\n" + "=" * 50)
    print("  Приклад 2. Адвективний перенос гаусового профілю")
    print("=" * 50)

    a = [1.0, 0.5]
    D = 0.01
    T = 0.4
    dt = 0.002
    nx = ny = 40

    def u0_func(x, y):
        return np.exp(-((x - 0.25)**2 + (y - 0.25)**2) / (2 * 0.08**2))

    g_func = lambda x, y, t: 0.0

    mesh = TriangularMesh(1.0, 1.0, nx, ny)
    M, K, C, _ = assemble_all(mesh, D, a)
    u0 = np.array([u0_func(x, y) for x, y in mesh.nodes])

    Pe_h = compute_peclet_number(a, mesh.hmax(), D)
    print(f"  a = {a}, D = {D}, Pe_h = {Pe_h:.1f}")
    print(f"  Сітка: {nx}x{ny}, вузлів: {mesh.n_nodes}")

    solver = Solver(mesh, M, K, C, dt, theta=1.0)
    solutions, times = solver.solve(u0, T, g_func=g_func, store_every=20)

    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    n = len(solutions)
    idx_list = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    fig, axes = plt.subplots(1, len(idx_list), figsize=(20, 4))
    vmin = min(np.min(solutions[i]) for i in idx_list)
    vmax = max(np.max(solutions[i]) for i in idx_list)
    levels = np.linspace(vmin, vmax, 20)

    for ax, idx in zip(axes, idx_list):
        tcf = ax.tricontourf(triang, solutions[idx],
                             levels=levels, cmap='inferno', extend='both')
        ax.set_title(f't = {times[idx]:.3f}', fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.colorbar(tcf, ax=axes, shrink=0.8, label='u')
    fig.suptitle(f'Приклад 2. Адвекція + дифузія (a = {a}, D = {D}, Pe_h = {Pe_h:.1f})',
                 fontsize=13, y=1.02)
    fig.savefig(os.path.join(OUTDIR, "example_2_transport.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {OUTDIR}/example_2_transport.png")


def example_3_galerkin_vs_supg():
    print("\n" + "=" * 50)
    print("  Приклад 3. Гальоркін vs SUPG (великі числа Пекле)")
    print("=" * 50)

    a = [2.0, 1.0]
    D = 0.001
    T = 0.3
    dt = 0.002
    nx = ny = 40

    def u0_func(x, y):
        return np.exp(-((x - 0.25)**2 + (y - 0.25)**2) / (2 * 0.08**2))

    g_func = lambda x, y, t: 0.0

    mesh = TriangularMesh(1.0, 1.0, nx, ny)
    Pe_h = compute_peclet_number(a, mesh.hmax(), D)
    print(f"  a = {a}, D = {D}, Pe_h = {Pe_h:.1f}")
    print(f"  Сітка: {nx}x{ny}")

    M, K, C, _ = assemble_all(mesh, D, a)
    u0 = np.array([u0_func(x, y) for x, y in mesh.nodes])

    solver_gal = Solver(mesh, M, K, C, dt, theta=1.0)
    sol_gal, times_gal = solver_gal.solve(u0, T, g_func=g_func, store_every=1000)

    M2, K2, C2, _ = assemble_all(mesh, D, a)
    S = assemble_supg_matrix(mesh, a, D)
    SM = assemble_supg_mass_matrix(mesh, a, D)

    solver_supg = Solver(mesh, M2 + SM, K2 + S, C2, dt, theta=1.0)
    sol_supg, times_supg = solver_supg.solve(u0, T, g_func=g_func, store_every=1000)

    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    all_data = np.concatenate([u0, sol_gal[-1], sol_supg[-1]])
    vmin, vmax = all_data.min(), all_data.max()
    levels = np.linspace(vmin, vmax, 20)

    ax = axes[0]
    ax.tricontourf(triang, u0, levels=levels, cmap='RdBu_r', extend='both')
    ax.set_title('Початкова умова, t=0', fontsize=11)
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = axes[1]
    ax.tricontourf(triang, sol_gal[-1], levels=levels, cmap='RdBu_r', extend='both')
    umin_g, umax_g = sol_gal[-1].min(), sol_gal[-1].max()
    ax.set_title(f'Гальоркін, t={times_gal[-1]:.2f}\nmin={umin_g:.4f}, max={umax_g:.4f}', fontsize=10)
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = axes[2]
    tcf2 = ax.tricontourf(triang, sol_supg[-1], levels=levels, cmap='RdBu_r', extend='both')
    umin_s, umax_s = sol_supg[-1].min(), sol_supg[-1].max()
    ax.set_title(f'SUPG, t={times_supg[-1]:.2f}\nmin={umin_s:.4f}, max={umax_s:.4f}', fontsize=10)
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

    fig.colorbar(tcf2, ax=axes, shrink=0.85, label='u')
    fig.suptitle(f'Приклад 3. Гальоркін vs SUPG (Pe_h = {Pe_h:.0f})',
                 fontsize=13, y=1.02)
    fig.savefig(os.path.join(OUTDIR, "example_3_galerkin_vs_supg.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {OUTDIR}/example_3_galerkin_vs_supg.png")


def example_4_source_term():
    print("\n" + "=" * 50)
    print("  Приклад 4. Стаціонарна задача Пуассона з джерелом")
    print("=" * 50)

    D = 1.0
    nx = ny = 40

    f_func = lambda x, y, t: 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    u_exact = lambda x, y, t: np.sin(np.pi * x) * np.sin(np.pi * y)
    g_func = lambda x, y, t: 0.0

    mesh = TriangularMesh(1.0, 1.0, nx, ny)
    M, K, C, _ = assemble_all(mesh, D, [0.0, 0.0])

    solver = Solver(mesh, M, K, C, dt=1.0, theta=1.0)
    u_h = solver.solve_stationary(f_func=f_func, g_func=g_func)

    err = l2_error(mesh, u_h, u_exact)
    u_ex = np.array([u_exact(x, y, 0) for x, y in mesh.nodes])
    print(f"  Сітка: {nx}x{ny}, вузлів: {mesh.n_nodes}")
    print(f"  L2-похибка: {err:.4e}")

    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    vmin = min(u_h.min(), u_ex.min())
    vmax = max(u_h.max(), u_ex.max())
    levels = np.linspace(vmin, vmax, 20)

    tcf0 = axes[0].tricontourf(triang, u_h, levels=levels, cmap='viridis')
    axes[0].set_title('Чисельний розв\'язок', fontsize=11)
    axes[0].set_aspect('equal'); axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

    axes[1].tricontourf(triang, u_ex, levels=levels, cmap='viridis')
    axes[1].set_title('Точний розв\'язок', fontsize=11)
    axes[1].set_aspect('equal'); axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

    err_field = np.abs(u_h - u_ex)
    tcf2 = axes[2].tricontourf(triang, err_field, levels=20, cmap='hot')
    axes[2].set_title(f'|Похибка|, L2={err:.2e}', fontsize=11)
    axes[2].set_aspect('equal'); axes[2].set_xlabel('x'); axes[2].set_ylabel('y')

    fig.colorbar(tcf0, ax=axes[:2], shrink=0.8, label='u')
    fig.colorbar(tcf2, ax=axes[2], shrink=0.8, label='|error|')
    fig.suptitle('Приклад 4. -Du = f, u = 0 на межі', fontsize=13, y=1.02)
    fig.savefig(os.path.join(OUTDIR, "example_4_source.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {OUTDIR}/example_4_source.png")


def example_5_neumann():
    print("\n" + "=" * 50)
    print("  Приклад 5. Гранична умова Неймана")
    print("=" * 50)

    D = 0.1
    T = 1.0
    dt = 0.005
    nx = ny = 32
    q0 = 1.0

    u0_func = lambda x, y: 0.0
    g_func = lambda x, y, t: 0.0
    q_func = lambda x, y, t: q0

    mesh = TriangularMesh(1.0, 1.0, nx, ny)
    M, K, C, _ = assemble_all(mesh, D, [0.0, 0.0])
    u0 = np.array([u0_func(x, y) for x, y in mesh.nodes])

    solver = Solver(mesh, M, K, C, dt, theta=1.0)
    solutions, times = solver.solve(
        u0, T, g_func=g_func,
        bc_sides=['left'],
        neumann_func=q_func,
        neumann_sides=['right', 'top', 'bottom'],
        store_every=20,
    )

    print(f"  D={D}, q0={q0}, T={T}")
    print(f"  Сітка: {nx}x{nx}, вузлів: {mesh.n_nodes}")
    print(f"  u_max при t={times[-1]:.2f}: {solutions[-1].max():.4f}")

    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    n = len(solutions)
    idx_list = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    fig, axes = plt.subplots(1, len(idx_list), figsize=(20, 4))
    vmin = min(np.min(solutions[i]) for i in idx_list)
    vmax = max(np.max(solutions[i]) for i in idx_list)
    levels = np.linspace(vmin, max(vmax, 0.01), 20)

    for ax, idx in zip(axes, idx_list):
        tcf = ax.tricontourf(triang, solutions[idx], levels=levels,
                             cmap='inferno', extend='both')
        ax.set_title(f't = {times[idx]:.3f}', fontsize=11)
        ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

    fig.colorbar(tcf, ax=axes, shrink=0.8, label='u')
    fig.suptitle(f'Приклад 5. Нейман (q={q0} на x=1, u=0 на x=0)',
                 fontsize=13, y=1.02)
    fig.savefig(os.path.join(OUTDIR, "example_5_neumann.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {OUTDIR}/example_5_neumann.png")


def example_6_robin():
    print("\n" + "=" * 50)
    print("  Приклад 6. Гранична умова Робіна")
    print("=" * 50)

    D = 0.05
    alpha = 5.0
    T = 1.0
    dt = 0.005
    nx = ny = 32

    u0_func = lambda x, y: 0.0
    g_func = lambda x, y, t: 1.0
    beta_func = lambda x, y, t: 0.0

    mesh = TriangularMesh(1.0, 1.0, nx, ny)
    M, K, C, _ = assemble_all(mesh, D, [0.0, 0.0])
    u0 = np.array([u0_func(x, y) for x, y in mesh.nodes])

    solver = Solver(mesh, M, K, C, dt, theta=1.0)
    solutions, times = solver.solve(
        u0, T, g_func=g_func,
        bc_sides=['left'],
        neumann_func=lambda x, y, t: 0.0,
        neumann_sides=['top', 'bottom'],
        robin_alpha=alpha, robin_beta=beta_func, robin_sides=['right'],
        store_every=20,
    )

    print(f"  D={D}, alpha={alpha}, T={T}")
    print(f"  Сітка: {nx}x{nx}, вузлів: {mesh.n_nodes}")
    print(f"  u(права стінка) при t={times[-1]:.2f}: "
          f"{solutions[-1][mesh.nodes[:, 0] > 0.99].mean():.4f}")

    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    n = len(solutions)
    idx_list = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    fig, axes = plt.subplots(1, len(idx_list), figsize=(20, 4))
    levels = np.linspace(0, 1, 20)

    for ax, idx in zip(axes, idx_list):
        tcf = ax.tricontourf(triang, solutions[idx], levels=levels,
                             cmap='coolwarm', extend='both')
        ax.set_title(f't = {times[idx]:.3f}', fontsize=11)
        ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

    fig.colorbar(tcf, ax=axes, shrink=0.8, label='u')
    fig.suptitle(f'Приклад 6. Робін (alpha={alpha}, D={D}, u=1 на x=0)',
                 fontsize=13, y=1.02)
    fig.savefig(os.path.join(OUTDIR, "example_6_robin.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {OUTDIR}/example_6_robin.png")


if __name__ == "__main__":
    example_1_diffusion()
    example_2_transport()
    example_3_galerkin_vs_supg()
    example_4_source_term()
    example_5_neumann()
    example_6_robin()
    print("\nГотово. Всі результати у папці results/")
