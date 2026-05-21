import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import io

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from PIL import Image

from src.assembly import assemble_all
from src.mesh import TriangularMesh
from src.solver import Solver
from src.stabilization import assemble_supg_mass_matrix, assemble_supg_matrix
from src.utils import compute_peclet_number

OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def gif_example_1():
    print("  GIF 1: Чиста дифузія...")
    D = 0.01; T = 0.5; dt = 0.005; nx = ny = 32

    u0_func = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    u_exact = lambda x, y, t: np.sin(np.pi*x)*np.sin(np.pi*y)*np.exp(-2*np.pi**2*D*t)
    g_func = lambda x, y, t: 0.0

    mesh = TriangularMesh(1.0, 1.0, nx, ny)
    M, K, C, _ = assemble_all(mesh, D, [0.0, 0.0])
    u0 = np.array([u0_func(x, y) for x, y in mesh.nodes])

    solver = Solver(mesh, M, K, C, dt, theta=1.0)
    solutions, times = solver.solve(u0, T, g_func=g_func, store_every=5)

    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)

    frames = []
    for i in range(len(solutions)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        u_ex = np.array([u_exact(x, y, times[i]) for x, y in mesh.nodes])

        tcf1 = ax1.tricontourf(triang, solutions[i], levels=np.linspace(0, 1, 15), cmap='viridis')
        ax1.set_title(f'Чисельний, t = {times[i]:.3f}', fontsize=11)
        ax1.set_aspect('equal'); ax1.set_xlabel('x'); ax1.set_ylabel('y')

        tcf2 = ax2.tricontourf(triang, u_ex, levels=np.linspace(0, 1, 15), cmap='viridis')
        ax2.set_title(f'Точний, t = {times[i]:.3f}', fontsize=11)
        ax2.set_aspect('equal'); ax2.set_xlabel('x'); ax2.set_ylabel('y')

        fig.suptitle('Приклад 1: Чиста дифузія (D = 0.01)', fontsize=13, y=1.02)
        fig.colorbar(tcf1, ax=[ax1, ax2], shrink=0.8, label='u')

        frames.append(fig_to_image(fig))
        plt.close(fig)

    path = os.path.join(OUTDIR, "example_1_diffusion.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=150, loop=0)
    print(f"    -> {path} ({len(frames)} кадрів)")


def gif_example_2():
    print("  GIF 2: Перенос горба...")
    a = [1.0, 0.5]; D = 0.01; T = 0.4; dt = 0.002; nx = ny = 40

    def u0_func(x, y):
        return np.exp(-((x - 0.25)**2 + (y - 0.25)**2) / (2 * 0.08**2))
    g_func = lambda x, y, t: 0.0

    mesh = TriangularMesh(1.0, 1.0, nx, ny)
    M, K, C, _ = assemble_all(mesh, D, a)
    u0 = np.array([u0_func(x, y) for x, y in mesh.nodes])

    solver = Solver(mesh, M, K, C, dt, theta=1.0)
    solutions, times = solver.solve(u0, T, g_func=g_func, store_every=10)

    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    vmax = max(np.max(s) for s in solutions)

    frames = []
    for i in range(len(solutions)):
        fig, ax = plt.subplots(figsize=(6, 5))
        tcf = ax.tricontourf(triang, solutions[i],
                             levels=np.linspace(0, vmax, 20), cmap='inferno', extend='both')
        fig.colorbar(tcf, ax=ax, shrink=0.85, label='u')
        ax.set_title(f'Перенос горба, t = {times[i]:.3f}\n'
                     f'a = {a}, D = {D}, Pe_h = {compute_peclet_number(a, mesh.hmax(), D):.1f}',
                     fontsize=11)
        ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

        frames.append(fig_to_image(fig))
        plt.close(fig)

    path = os.path.join(OUTDIR, "example_2_transport.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=150, loop=0)
    print(f"    -> {path} ({len(frames)} кадрів)")


def gif_example_3():
    print("  GIF 3: Гальоркін vs SUPG...")
    a = [2.0, 1.0]; D = 0.001; T = 0.3; dt = 0.002; nx = ny = 40

    def u0_func(x, y):
        return np.exp(-((x - 0.25)**2 + (y - 0.25)**2) / (2 * 0.08**2))
    g_func = lambda x, y, t: 0.0

    mesh = TriangularMesh(1.0, 1.0, nx, ny)

    M, K, C, _ = assemble_all(mesh, D, a)
    u0 = np.array([u0_func(x, y) for x, y in mesh.nodes])
    solver_gal = Solver(mesh, M, K, C, dt, theta=1.0)
    sol_gal, times_gal = solver_gal.solve(u0, T, g_func=g_func, store_every=10)

    M2, K2, C2, _ = assemble_all(mesh, D, a)
    S = assemble_supg_matrix(mesh, a, D)
    SM = assemble_supg_mass_matrix(mesh, a, D)
    solver_supg = Solver(mesh, M2 + SM, K2 + S, C2, dt, theta=1.0)
    sol_supg, times_supg = solver_supg.solve(u0, T, g_func=g_func, store_every=10)

    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    n = min(len(sol_gal), len(sol_supg))

    all_vals = np.concatenate([np.concatenate(sol_gal[:n]), np.concatenate(sol_supg[:n])])
    vmin, vmax = all_vals.min(), all_vals.max()
    levels = np.linspace(vmin, vmax, 20)

    frames = []
    for i in range(n):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        ax1.tricontourf(triang, sol_gal[i], levels=levels, cmap='RdBu_r', extend='both')
        g_min, g_max = sol_gal[i].min(), sol_gal[i].max()
        ax1.set_title(f'Гальоркін, t={times_gal[i]:.3f}\n'
                      f'min={g_min:.4f}, max={g_max:.4f}', fontsize=10)
        ax1.set_aspect('equal'); ax1.set_xlabel('x'); ax1.set_ylabel('y')

        tcf = ax2.tricontourf(triang, sol_supg[i], levels=levels, cmap='RdBu_r', extend='both')
        s_min, s_max = sol_supg[i].min(), sol_supg[i].max()
        ax2.set_title(f'SUPG, t={times_supg[i]:.3f}\n'
                      f'min={s_min:.4f}, max={s_max:.4f}', fontsize=10)
        ax2.set_aspect('equal'); ax2.set_xlabel('x'); ax2.set_ylabel('y')

        fig.colorbar(tcf, ax=[ax1, ax2], shrink=0.85, label='u')
        Pe_h = compute_peclet_number(a, mesh.hmax(), D)
        fig.suptitle(f'Гальоркін vs SUPG (Pe_h = {Pe_h:.0f})', fontsize=13, y=1.02)

        frames.append(fig_to_image(fig))
        plt.close(fig)

    path = os.path.join(OUTDIR, "example_3_galerkin_vs_supg.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=200, loop=0)
    print(f"    -> {path} ({len(frames)} кадрів)")


if __name__ == "__main__":
    print("Генерація GIF-анімацій...")
    gif_example_1()
    gif_example_2()
    gif_example_3()
    print("Готово.")
