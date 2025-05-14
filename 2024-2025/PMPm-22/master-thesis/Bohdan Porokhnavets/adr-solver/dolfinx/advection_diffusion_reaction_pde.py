"""
Advection-diffusion-reaction pde:

  du/dt - D(dot(grad(u), grad(u))) + dot(v, grad(u)) + Ku = f

"""

import matplotlib.pyplot as plt
from fenics import Expression
import numpy as np

from dolfinx.fem import dirichletbc, functionspace, locate_dofs_geometrical
from dolfinx.fem import Function, Constant
from dolfinx.mesh import create_rectangle, CellType

from mpi4py import MPI
from petsc4py import PETSc

import pyvista
from dolfinx import fem, plot
from ufl import TrialFunction, TestFunction, lhs, rhs
from ufl import dx, grad, dot, inner
from dolfinx import mesh, fem
from adr_types import AdrParams, Velocity, Rectangle, exact, on_boundary


from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, set_bc
from ufl import dx, grad, dot, inner

import logging
import dolfin.cpp.log as dolfin_log
dolfin_log.set_log_level(logging.ERROR)

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
    print(f"  u_exact = {adrParams.u_exact.label()}")

    p0 = [0.0, 0.0]
    p1 = [adrParams.plane.x, adrParams.plane.y]

    # mesh = create_rectangle(MPI.COMM_WORLD, [p0, p1], [adrParams.N_mesh, adrParams.N_mesh], CellType.triangle)
    mesh = create_rectangle(MPI.COMM_WORLD, [p0, p1], [adrParams.N_mesh, adrParams.N_mesh], CellType.quadrilateral)
    # print("Geometry dim:", mesh.geometry.dim)
    # print("Topology dim:", mesh.topology.dim)
    # print("Number of cells:", mesh.topology.index_map(mesh.topology.dim).size_local)
    # print("Number of vertices:", mesh.topology.index_map(0).size_local)

    V = functionspace(mesh, ("Lagrange", 3))
    u = TrialFunction(V)
    v = TestFunction(V)

    uex = Function(V)
    uex.interpolate(lambda x: exact(x, adrParams.u_exact))
    def update_exec(t):
        adrParams.u_exact.t = t
        uex.interpolate(lambda x: exact(x, adrParams.u_exact))

    # Boundary conditions
    boundary_dofs = locate_dofs_geometrical(V, lambda x: on_boundary(x, 0.0, adrParams.plane.x, 0.0, adrParams.plane.y))
    bc = dirichletbc(uex, boundary_dofs)

    def plot_boundary():
        # Create a plot of the solution used for BC
        grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
        grid.point_data["uex"] = uex.x.array  # or any function values
        grid.point_data["x"] = grid.points[:, 0]
        grid.point_data["y"] = grid.points[:, 1]

        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, scalars="uex", show_edges=True, cmap="viridis")
        plotter.add_point_labels(grid.points,  # Optional: show coordinates as labels
                                labels=[f"x={x:.2f}, y={y:.2f}" for x, y in grid.points[:, :2]],
                                font_size=10, point_size=5, render_points_as_spheres=True)

        points = grid.points
        values = uex.x.array

        for i, (x, y) in enumerate(points[:, :2]):  # 2D case
            print(f"Point {i}: x = {x:.3f}, y = {y:.3f}, uex = {values[i]:.5f}")
        for i, (x, y) in enumerate(points[:, :2]):  # 2D case
            if i in boundary_dofs:
                print(f"Boundary Point {i}: x = {x:.3f}, y = {y:.3f}, uex = {values[i]:.5f}")

        print(f"Location of uex at the boundary: {len(boundary_dofs)}")
        print(f"Values of uex at the boundary: {len(uex.x.array[boundary_dofs])}")
        print(f"Location of uex at the boundary: {boundary_dofs}")
        print(f"Values of uex at the boundary: {uex.x.array[boundary_dofs]}")
        # plotter.show()

    plot_boundary()

    dt = adrParams.T_time / adrParams.M_time_iter

    k = Constant(mesh.ufl_domain(), PETSc.ScalarType(dt))
    Diffution = Constant(mesh.ufl_domain(), PETSc.ScalarType(adrParams.diffusion))
    react = Constant(mesh.ufl_domain(), PETSc.ScalarType(adrParams.reaction))
    beta = Constant(mesh.ufl_domain(), np.array([adrParams.velocity.x, adrParams.velocity.y]))
    f = Function(V)
    def update_source(t):
        adrParams.f_source.t = t
        f.interpolate(lambda x: exact(x, adrParams.f_source))
    update_source(0)

    u_n = Function(V)
    u_n.interpolate(lambda x: exact(x, adrParams.u_exact))

    errors_list = []

    for step in range(adrParams.M_time_iter):
        t = (step + 1) * dt
        update_source(t)
        update_exec(t)


        # Update boundary and source
        bc = dirichletbc(uex, boundary_dofs)

        # Weak forms
        a = (1/k)*u*v*dx + Diffution*inner(grad(u), grad(v))*dx + inner(beta, grad(u))*v*dx + react*u*v*dx
        L = (1/k)*u_n*v*dx + f*v*dx
        a_form = fem.form(a)
        L_form = fem.form(L)

        # Assemble system
        A = assemble_matrix(a_form, bcs=[bc])
        A.assemble()
        b = assemble_vector(L_form)
        set_bc(b, [bc])

        # Solve system
        x = create_vector(L_form)
        x.set(0)
        solver = PETSc.KSP().create(mesh.comm)
        solver.setOperators(A)
        solver.setType("cg")
        solver.getPC().setType("hypre")
        solver.solve(b, x)

        # Update solution
        u_n.x.array[:] = x.array
        u_n.x.scatter_forward()
        assert np.all(np.isfinite(u_n.x.array))

        error_L2 = np.sqrt(fem.assemble_scalar(fem.form((u_n - uex)**2 * dx)))

        errors = []
        for x_ex, x_new in zip(uex.x.array, u_n.x.array):
            # print(f"ex = {x_ex}, sol = {x_new}, diff = {abs(x_ex - x_new)}")
            errors.append(abs(x_ex - x_new))

        def plot_boundary_res():
        
            # Create a plot of the solution used for BC
            grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
            grid.point_data["u_n"] = errors  # or any function values
            grid.point_data["x"] = grid.points[:, 0]
            grid.point_data["y"] = grid.points[:, 1]

            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, scalars="u_n", show_edges=True, cmap="viridis")
            plotter.add_point_labels(grid.points,  # Optional: show coordinates as labels
                                    labels=[f"x={x:.2f}, y={y:.2f}" for x, y in grid.points[:, :2]],
                                    font_size=10, point_size=5, render_points_as_spheres=True)

            points = grid.points
            values = errors

            for i, (x, y) in enumerate(points[:, :2]):  # 2D case
                print(f"Res Point {i}: x = {x:.3f}, y = {y:.3f}, uex = {values[i]:.5f}")
            for i, (x, y) in enumerate(points[:, :2]):  # 2D case
                if i in boundary_dofs:
                    print(f"Res Boundary Point {i}: x = {x:.3f}, y = {y:.3f}, uex = {values[i]:.5f}")
            plotter.show()

        # plot_boundary_res()

        print(f"Res {error_L2}")
        print(f"Res {np.sum(errors)/len(errors)}")

        errors_list.append(error_L2)


    for er in errors_list:
        print(er)

def main():
    print("adr main")
    plane=Rectangle(2, 2)
    N_mesh=80
    T_time=1.0
    M_time_iter=10

    # du/dt - 0.001(dot(grad(u), grad(u))) + dot([1,1], grad(u)) + u = 0
    # u_exact = e^((1/2)*(0.001*t - x - y))
    adr_test_params1 = AdrParams(
        diffusion=0.001, 
        reaction=1.0,
        velocity=Velocity(1.0, 1.0), 
        f_source=Expression('0', degree=3, t=0),
        plane=plane,
        N_mesh=N_mesh,
        T_time=T_time,
        M_time_iter=M_time_iter,
        u_exact=Expression('exp((D*t -x[0] - x[1])/2)', degree=3, D=0.001, t=0)
    )

    # du/dt - 0.001(dot(grad(u), grad(u))) + dot([1,2], grad(u)) + 2u = (1/2)*e^((1/2)*(0.001*t - x - y))
    # u_exact = e^((1/2)*(0.001*t - x - y))
    adr_test_params2 = AdrParams( 
        diffusion=0.1, 
        reaction=2.0,
        velocity=Velocity(1.0, 2.0), 
        f_source=Expression('exp((D*t -x[0] - x[1])/2)/2', degree=5, D=0.1, t=0),
        plane=plane,
        N_mesh=N_mesh,
        T_time=T_time,
        M_time_iter=M_time_iter,
        u_exact=Expression('exp((D*t -x[0] - x[1])/2)', degree=5, D=0.1, t=0)
    )

    adr_test_params3 = AdrParams(
        diffusion=0.1, 
        reaction=1.0,
        velocity=Velocity(1.0, 0.0),
        f_source=Expression('0', degree=3, alpha=0.001, t=0),
        plane=plane,
        N_mesh=N_mesh,
        T_time=T_time,
        M_time_iter=M_time_iter,
        u_exact=Expression('exp((D*t -x[0] - x[1])/2)', degree=3, D=0.1, t=0)
    )
    solve_adr_pde(adr_test_params1)
    

if __name__ == "__main__":
    main()
