import time
import os
import numpy as np
import matplotlib.pyplot as plt
from fenics import *

# Вимикаємо зайвий спам у консоль
set_log_level(LogLevel.ERROR)


def advanced_stokes(N=128):
    print("\n=== 1. МОДЕЛЮВАННЯ ТЕЧІЇ В КАВЕРНІ ===")
    print(f"Сітка: {N}x{N} елементів")
    start_time = time.time()

    if not os.path.exists("advanced_output"):
        os.makedirs("advanced_output")

    mesh = UnitSquareMesh(N, N)
    V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V * Q)

    def top_boundary(x, on_boundary): return on_boundary and near(x[1], 1.0)

    def walls_boundary(x, on_boundary): return on_boundary and (near(x[0], 0.0) or near(x[0], 1.0) or near(x[1], 0.0))

    def origin_point(x, on_boundary): return near(x[0], 0.0) and near(x[1], 0.0)

    bc_top = DirichletBC(W.sub(0), Constant((1.0, 0.0)), top_boundary)
    bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls_boundary)
    bc_p = DirichletBC(W.sub(1), Constant(0.0), origin_point, "pointwise")
    bcs = [bc_top, bc_walls, bc_p]

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    f = Constant((0.0, 0.0))

    a = (inner(grad(u), grad(v)) - p * div(v) + q * div(u)) * dx
    L = inner(f, v) * dx

    w = Function(W)
    solve(a == L, w, bcs, solver_parameters={'linear_solver': 'mumps'})

    u_sol, p_sol = w.split()
    u_sol.rename("Velocity", "velocity")
    p_sol.rename("Pressure", "pressure")

    div_u = project(div(u_sol), FunctionSpace(mesh, "CG", 1))
    div_norm = norm(div_u, 'L2')
    print(f"Похибка нестисливості: {div_norm:.3e}")

    with XDMFFile("advanced_output/stokes_results.xdmf") as xdmf:
        xdmf.parameters["flush_output"] = True
        xdmf.parameters["functions_share_mesh"] = True
        xdmf.write(u_sol, 0)
        xdmf.write(p_sol, 0)

    y_vals = np.linspace(0, 1, 200)
    u_x_profile = [u_sol(0.5, y)[0] for y in y_vals]
    np.savetxt("advanced_output/center_profile.csv",
               np.column_stack((y_vals, u_x_profile)),
               delimiter=",", header="y,u_x", comments="")

    print(f"Готово! Час: {time.time() - start_time:.2f} сек.")


def run_convergence_study():
    print("\n=== 2. ДОСЛІДЖЕННЯ ЗБІЖНОСТІ (MMS) ===")
    mesh_sizes = [16, 32, 64, 128]
    h_vals, error_u, error_p = [], [], []

    for N in mesh_sizes:
        mesh = UnitSquareMesh(N, N)
        h_vals.append(1.0 / N)

        V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
        Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        W = FunctionSpace(mesh, V * Q)

        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)

        u_ex = Expression(("sin(pi*x[1])", "cos(pi*x[0])"), degree=4)
        p_ex = Expression("sin(pi*x[0])*cos(pi*x[1])", degree=4)
        f = Expression(("pi*pi*sin(pi*x[1]) + pi*cos(pi*x[0])*cos(pi*x[1])",
                        "pi*pi*cos(pi*x[0]) - pi*sin(pi*x[0])*sin(pi*x[1])"), degree=4)

        a = (inner(grad(u), grad(v)) - p * div(v) + q * div(u)) * dx
        L = inner(f, v) * dx

        def bnd(x, on_boundary): return on_boundary

        bc = DirichletBC(W.sub(0), u_ex, bnd)

        w = Function(W)
        solve(a == L, w, bc, solver_parameters={'linear_solver': 'mumps'})
        u_sol, p_sol = w.split()

        err_u = errornorm(u_ex, u_sol, 'L2')
        err_p = errornorm(p_ex, p_sol, 'L2')

        error_u.append(err_u)
        error_p.append(err_p)
        print(f"N={N:3d} | Err(u)={err_u:.2e} | Err(p)={err_p:.2e}")

    plt.figure(figsize=(8, 6))
    plt.loglog(h_vals, error_u, 'o-', linewidth=2, label=r'$L_2$ похибка $\mathbf{u}$')
    plt.loglog(h_vals, error_p, 's-', linewidth=2, label=r'$L_2$ похибка $p$')

    h_ref = np.array(h_vals)
    plt.loglog(h_vals, 10 * h_ref ** 3, 'k--', label='$O(h^3)$')
    plt.loglog(h_vals, 5 * h_ref ** 2, 'k:', label='$O(h^2)$')

    plt.xlabel('Розмір кроку сітки $h$', fontsize=12)
    plt.ylabel('Норма похибки $L_2$', fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=12)
    plt.savefig('convergence.png', dpi=300, bbox_inches='tight')
    print("Графік збережено як convergence.png")


def run_channel_flow():
    print("\n=== 3. ОБТІКАННЯ КАНАЛУ (Channel Flow) ===")
    start_time = time.time()

    # Геометрія: простий прямокутник (без mshr)
    mesh = RectangleMesh(Point(0, 0), Point(2.2, 0.41), 160, 30)

    V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V * Q)

    # Профіль входу
    inflow_profile = Expression(('4*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0'), degree=2)

    def inflow(x, on_boundary): return on_boundary and near(x[0], 0.0)

    def walls(x, on_boundary): return on_boundary and (near(x[1], 0) or near(x[1], 0.41))

    def outflow(x, on_boundary): return on_boundary and near(x[0], 2.2)

    bc_in = DirichletBC(W.sub(0), inflow_profile, inflow)
    bc_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls)
    bc_out_p = DirichletBC(W.sub(1), Constant(0), outflow)
    bcs = [bc_in, bc_walls, bc_out_p]

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    f = Constant((0, 0))

    mu = 0.001
    a = (mu * inner(grad(u), grad(v)) - p * div(v) + q * div(u)) * dx
    L = inner(f, v) * dx

    w = Function(W)
    solve(a == L, w, bcs, solver_parameters={'linear_solver': 'mumps'})
    u_sol, p_sol = w.split()

    with XDMFFile("advanced_output/channel_results.xdmf") as xdmf:
        xdmf.write(u_sol, 0)

    print(f"Готово! Час: {time.time() - start_time:.2f} сек.")
    print("Результати збережено у advanced_output/channel_results.xdmf")


if __name__ == "__main__":
    advanced_stokes(N=128)
    run_convergence_study()
    run_channel_flow()