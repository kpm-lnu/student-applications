"""
Advection-diffusion-reaction pde:

  du/dt - D(dot(grad(u), grad(u))) + dot(v, grad(u)) + Ku = f

"""


import matplotlib.pyplot as plt
from fenics import *
import numpy as np
from dolfin import near

from adr_types import AdrParams, Velocity, Rectangle, paraview_show_results

import logging
import dolfin.cpp.log as dolfin_log
dolfin_log.set_log_level(logging.ERROR)

PARAVIEW_RESULT_FILE_NAME = 'adr-solver-results/adr-solution.pvd'

# Example 1:
# du/dt - 0.001(dot(grad(u), grad(u))) + dot([1,1], grad(u)) + u = 0
# u_exact = e^((1/2)*(0.001*t - x - y))
# -----------------------------------------------------------------------------
# D = Constant(0.001)  # Diffusion
# K = Constant(1.0)    # Reaction
# v_x = Constant(1.0)  # Velocity x
# v_y = Constant(1.0)  # Velocity y
# b = Expression((('v_x', 'v_y')), v_x=v_x, v_y=v_y, degree=4, t=0)
# f = Expression('0', degree=4, alpha=D, t=0)
# u_exact = Expression('exp((alpha*t -x[0] - x[1])/2)', degree=4, alpha=D, t=0)
# -----------------------------------------------------------------------------


# Example 2:
# du/dt - 0.1(dot(grad(u), grad(u))) + dot([1,2], grad(u)) + 2u = (1/2)*e^((1/2)*(0.1*t - x - y))
# u_exact = e^((1/2)*(0.1*t - x - y))
# # -----------------------------------------------------------------------------
# D = Constant(0.1)    # Diffusion
# K = Constant(2.0)    # Reaction
# v_x = Constant(1.0)  # Velocity x
# v_y = Constant(2.0)  # Velocity y
# b = Expression((('v_x', 'v_y')), v_x=v_x, v_y=v_y, degree=4, t=0)
# f = Expression('exp((alpha*t -x[0] - x[1])/2)/2', degree=4, alpha=D, t=0)
# u_exact = Expression('exp((alpha*t -x[0] - x[1])/2)', degree=4, alpha=D, t=0)
# -----------------------------------------------------------------------------

# Example 3:
# du/dt - 0.1(dot(grad(u), grad(u))) + dot([1,0], grad(u)) + u = (1/2)*e^((1/2)*(0.1*t - x - y))
# u_exact = e^((1/2)*(0.1*t - x - y))
# # -----------------------------------------------------------------------------
# D = Constant(0.1)    # Diffusion
# K = Constant(1.0)    # Reaction
# v_x = Constant(1.0)  # Velocity x
# v_y = Constant(0.0)  # Velocity y
# b = Expression((('v_x', 'v_y')), v_x=v_x, v_y=v_y, degree=4, t=0)
# f = Expression('exp((alpha*t -x[0] - x[1])/2)/2', degree=4, alpha=D, t=0)
# u_exact = Expression('exp((alpha*t -x[0] - x[1])/2)', degree=4, alpha=D, t=0)
# # -----------------------------------------------------------------------------


def boundary(x, on_boundary):
    return on_boundary

def solve_adr_pde(adrParams: AdrParams):
    print("solve_adr_pde")
    print(f"  D = {float(adrParams.diffusion)}")
    print(f"  K = {float(adrParams.reaction)}")
    print(f"  v_x = {float(adrParams.velocity.x)}")
    print(f"  v_y = {float(adrParams.velocity.y)}")
    print(f"  f = {adrParams.f_source.label()}")
    print(f"  N = {adrParams.N_mesh}")
    print(f"  T = {adrParams.T_time}")
    print(f"  M = {adrParams.M_time_iter}")
    print(f"  A x = {adrParams.plane.x}")
    print(f"  A y = {adrParams.plane.y}")
    print(f"  boundary x0 = {adrParams.boundary[0].label() if len(adrParams.boundary) > 0 else "None"}")
    print(f"  boundary x1 = {adrParams.boundary[1].label() if len(adrParams.boundary) > 1 else "None"}")
    print(f"  boundary y0 = {adrParams.boundary[2].label() if len(adrParams.boundary) > 2 else "None"}")
    print(f"  boundary y1 = {adrParams.boundary[3].label() if len(adrParams.boundary) > 3 else "None"}")
    print(f"  initial condition = {adrParams.initial.label() if adrParams.initial is not None else "None"}")
    print(f"  u_exact = {adrParams.u_exact.label()}")

    N = adrParams.N_mesh
    T = adrParams.T_time
    M = adrParams.M_time_iter

    D = adrParams.diffusion
    K = adrParams.reaction
    b = Expression((('v_x', 'v_y')), v_x=adrParams.velocity.x, v_y=adrParams.velocity.y, degree=5, t=0)
    f = adrParams.f_source
    u_exact = adrParams.u_exact

    # Create mesh and define function space
    mesh = RectangleMesh(Point(0, 0), Point(adrParams.plane.x, adrParams.plane.y), N, N)

    # Bubble
    # V = FunctionSpace(mesh, "B", 3)
    # Lagrange
    V = FunctionSpace(mesh, "P", 4)


    def left(x, on_boundary):
        return on_boundary and near(x[0], 0.0)
    def right(x, on_boundary):
        return on_boundary and near(x[0], adrParams.plane.x)
    def bottom(x, on_boundary):
        return on_boundary and near(x[1], 0.0)
    def top(x, on_boundary):
        return on_boundary and near(x[1], adrParams.plane.y)

    # Define boundary condition
    if len(adrParams.boundary) == 4:
        bc_left = DirichletBC(V, adrParams.boundary[0], left)
        bc_right = DirichletBC(V, adrParams.boundary[1], right)
        bc_bottom = DirichletBC(V, adrParams.boundary[2], bottom)
        bc_top = DirichletBC(V, adrParams.boundary[3], top)
        bc = [bc_left, bc_right, bc_bottom, bc_top]
    else:
        bc = DirichletBC(V, u_exact, boundary)

    # Define initial value
    u_n = interpolate(u_exact if adrParams.initial is None else adrParams.initial, V)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    dt = T / M
    k = Constant(dt)

    # Main weak formulation
    F = ((u - u_n) / k)*v*dx + D*dot(grad(u), grad(v)) * \
        dx + (dot(grad(u), b) + K*u) * v*dx - f*v*dx

    a, L = lhs(F), rhs(F)

    u_solve = Function(V)
    t = 0

    errors = []
    vtkfile = File(PARAVIEW_RESULT_FILE_NAME)

    for n in range(M):
        # Update current time
        t += dt
        b.t = t
        f.t = t
        u_exact.t = t
        for bound in adrParams.boundary:
            bound.t = t

        u_v_exact = interpolate(u_exact, V)


        if len(adrParams.boundary) == 4:
            bc_left = DirichletBC(V, adrParams.boundary[0], left)
            bc_right = DirichletBC(V, adrParams.boundary[1], right)
            bc_bottom = DirichletBC(V, adrParams.boundary[2], bottom)
            bc_top = DirichletBC(V, adrParams.boundary[3], top)
            bc = [bc_left, bc_right, bc_bottom, bc_top]
        else:
            bc = DirichletBC(V, u_exact, boundary)

        solve(a == L, u_solve, bcs=bc)


        # Compute error
        error_t = errornorm(u_v_exact, u_solve, 'L2', mesh=mesh)
        errors.append(error_t)

        u_n.assign(u_solve)
        vtkfile << u_solve


    time_avg_error = np.sum(errors) / M
    print('Time-averaged L2 error:', time_avg_error)
    print('Time-max L2 error:', np.max(errors))

    # Hold plot
    plt.plot(np.linspace(0, T, M), errors)
    plt.xlabel('Time')
    plt.ylabel('L2 Error')
    plt.show()

    paraview_show_results(PARAVIEW_RESULT_FILE_NAME)


def main():
    plane=Rectangle(4, 4)
    N_mesh=40
    T_time=10.0
    M_time_iter=100

    # Example 1:
    # du/dt - 0.001(dot(grad(u), grad(u))) + dot([1,1], grad(u)) + u = 0
    # u_exact = e^((1/2)*(0.001*t - x - y))
    adr_test_params1 = AdrParams(
        diffusion=0.001, 
        reaction=1.0,
        velocity=Velocity(1.0, 1.0), 
        f_source=Expression('0', degree=4, alpha=0.001, t=0),
        plane=plane,
        N_mesh=N_mesh,
        T_time=T_time,
        M_time_iter=M_time_iter,
        boundary=[],
        initial=None,
        u_exact=Expression('exp((D*t -x[0] - x[1])/2)', degree=4, D=0.001, t=0)
    )

    # Example 2:
    # du/dt - 0.1(dot(grad(u), grad(u))) + dot([1,2], grad(u)) + 2u = (1/2)*e^((1/2)*(0.1*t - x - y))
    # u_exact = e^((1/2)*(0.1*t - x - y))
    adr_test_params2 = AdrParams( 
        diffusion=0.1, 
        reaction=2.0,
        velocity=Velocity(1.0, 2.0), 
        f_source=Expression('exp((D*t -x[0] - x[1])/2)/2', degree=4, D=0.1, t=0),
        plane=plane,
        N_mesh=N_mesh,
        T_time=T_time,
        M_time_iter=M_time_iter,
        boundary=[],
        initial=None,
        u_exact=Expression('exp((D*t -x[0] - x[1])/2)', degree=4, D=0.1, t=0)
    )

    # Example 3:
    # du/dt - 0.1(dot(grad(u), grad(u))) + dot([1,0], grad(u)) + u = (1/2)*e^((1/2)*(0.1*t - x - y))
    # u_exact = e^((1/2)*(0.1*t - x - y))
    adr_test_params3 = AdrParams(
        diffusion=0.1, 
        reaction=1.0,
        velocity=Velocity(1.0, 0.0),
        f_source=Expression('exp((D*t -x[0] - x[1])/2)/2', degree=4, D=0.1, t=0),
        plane=plane,
        N_mesh=N_mesh,
        T_time=T_time,
        M_time_iter=M_time_iter,
        boundary=[],
        initial=None,
        u_exact=Expression('exp((D*t -x[0] - x[1])/2)', degree=4, D=0.1, t=0)
    )

    solve_adr_pde(adr_test_params2)

if __name__ == "__main__":
    main()
