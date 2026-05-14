"""
FEniCSx FEM model: 2D reaction-diffusion problem with a 1D internal inclusion.

- Omega is a 2D unit square.
- Gamma is an internal 1D line y = y_line.
- The effect of Gamma is included through additional integrals over internal facets dS(1).

Two modes are available:
1) manufactured:
   Uses an exact solution and computes L2 errors in Omega and on Gamma.
   Homogeneous Dirichlet boundary conditions are applied because the exact solution is zero on the boundary.

2) demo:
   Uses a Gaussian initial condition near the top of the domain.
   No Dirichlet boundary conditions are applied. By default beta_boundary=0, so the problem has natural zero-flux boundaries.
   If beta_boundary > 0, a Robin boundary leakage term D*du/dn + beta*u = 0 is used.

Run in the FEniCSx Docker container, for example:
    python3 fenicsx_1d_inclusion_final.py --mode manufactured --output RESULT
    python3 fenicsx_1d_inclusion_final.py --mode demo --output RESULT
"""

from mpi4py import MPI
from petsc4py import PETSc

import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import ufl
from dolfinx import mesh, fem, plot
from dolfinx.fem.petsc import LinearProblem


# ============================================================
# Arguments
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="FEniCSx FEM: 2D reaction-diffusion with a 1D internal inclusion"
    )

    parser.add_argument("--mode", choices=["manufactured", "demo"], default="manufactured")

    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)

    parser.add_argument("--T", type=float, default=2)
    parser.add_argument("--steps", type=int, default=100)

    parser.add_argument("--y-line", type=float, default=0.5)

    # 2D medium coefficients
    parser.add_argument("--D-omega", type=float, default=0.05)
    parser.add_argument("--c-omega", type=float, default=0.03)

    # Boundary leakage coefficient for Robin boundary condition in demo mode
    parser.add_argument("--beta-boundary", type=float, default=0.0,
                        help="Robin boundary leakage coefficient beta. beta=0 gives natural zero-flux boundary in demo mode")

    # 1D inclusion coefficients
    parser.add_argument("--alpha-gamma", type=float, default=1)
    parser.add_argument("--D-gamma", type=float, default=0.1)
    parser.add_argument("--c-gamma", type=float, default=0.9)

    # Manufactured solution shape
    parser.add_argument("--shift-y", type=float, default=1.0,
                        help="Shifts manufactured exact solution upward while preserving zero boundary values")

    # Demo Gaussian parameters
    parser.add_argument("--demo-x0", type=float, default=0.2)
    parser.add_argument("--demo-y0", type=float, default=0.6)
    parser.add_argument("--demo-sigma", type=float, default=0.075)

    parser.add_argument("--output", type=str, default="AMI_the_best")

    return parser.parse_args()


# ============================================================
# Manufactured exact solution
# ============================================================


def exact_np(x, t, shift_y):
    """Exact solution in NumPy form for interpolation and plotting."""
    return (
        np.exp(-t)
        * np.sin(np.pi * x[0])
        * np.sin(np.pi * x[1])
        * (1.0 + shift_y * (x[1] - 0.5))
    )


def exact_ufl(x, t, shift_y):
    """Exact solution in UFL form for manufactured source terms and errors."""
    return (
        ufl.exp(-t)
        * ufl.sin(np.pi * x[0])
        * ufl.sin(np.pi * x[1])
        * (1.0 + shift_y * (x[1] - 0.5))
    )


# ============================================================
# 1D inclusion marking: Gamma = {y = y_line}
# ============================================================


def create_internal_line_tags(domain, y_line):
    """
    Mark internal facets located on the horizontal line y = y_line.

    In 2D, facets are 1D edges. The line inclusion Gamma is therefore
    represented by a set of internal mesh edges and integrated using dS(1).
    """

    tdim = domain.topology.dim
    fdim = tdim - 1

    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(tdim, fdim)

    facets = mesh.locate_entities(
        domain,
        fdim,
        lambda x: np.isclose(x[1], y_line, atol=1e-12)
    )

    facets = np.asarray(facets, dtype=np.int32)

    if len(facets) == 0:
        raise RuntimeError(
            "No facets were found on the inclusion line. "
            "Use y_line that lies exactly on the mesh, e.g. y_line=0.5 and even ny."
        )

    values = np.full(len(facets), 1, dtype=np.int32)
    order = np.argsort(facets)

    facet_tags = mesh.meshtags(domain, fdim, facets[order], values[order])
    return facet_tags, facets


# ============================================================
# Plotting helpers
# ============================================================


def build_triangulation(V):
    topology, cell_types, geometry = plot.vtk_mesh(V)

    if topology.ndim == 1:
        cells = topology.reshape((-1, 4))[:, 1:4]
    else:
        if topology.shape[1] == 4:
            cells = topology[:, 1:4]
        else:
            cells = topology

    return tri.Triangulation(geometry[:, 0], geometry[:, 1], cells)


def save_field_plot(V, function, filename, title, y_line=None):
    triangulation = build_triangulation(V)
    values = function.x.array.real

    plt.figure(figsize=(6, 5))
    plt.tricontourf(triangulation, values, levels=60)
    plt.colorbar(label="u")

    if y_line is not None:
        plt.axhline(y_line, linestyle="--", linewidth=1.5, label="1D inclusion")
        plt.legend(loc="upper right")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def save_vertical_profile(V, function, filename, x0, title):
    coords = V.tabulate_dof_coordinates()[:, :2]
    values = function.x.array.real

    mask = np.isclose(coords[:, 0], x0, atol=1e-12)
    if not np.any(mask):
        print(f"Warning: no DOFs found on x={x0} for vertical profile")
        return

    y = coords[mask, 1]
    u = values[mask]
    order = np.argsort(y)

    plt.figure(figsize=(5, 4))
    plt.plot(y[order], u[order], marker="o", markersize=3)
    plt.xlabel("y")
    plt.ylabel("u")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def save_line_profile(V, numerical, exact_function, filename, y_line, time_value, shift_y, include_exact):
    coords = V.tabulate_dof_coordinates()[:, :2]
    numerical_values = numerical.x.array.real

    mask = np.isclose(coords[:, 1], y_line, atol=1e-12)
    if not np.any(mask):
        print("Warning: no DOFs found on the inclusion line")
        return

    x_line = coords[mask, 0]
    u_num = numerical_values[mask]
    order = np.argsort(x_line)

    x_line = x_line[order]
    u_num = u_num[order]

    plt.figure(figsize=(7, 4))
    plt.plot(x_line, u_num, marker="o", markersize=3, label="numerical trace on Gamma")

    if include_exact:
        points = np.zeros((2, len(x_line)))
        points[0, :] = x_line
        points[1, :] = y_line
        u_ex = exact_np(points, time_value, shift_y)
        plt.plot(x_line, u_ex, linewidth=2, label="exact trace")

    plt.xlabel("x along Gamma")
    plt.ylabel("u")
    plt.title("Solution trace on the 1D inclusion")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def save_error_history(times, err_omega, err_gamma, err_total, rel_total, filename):
    plt.figure(figsize=(7, 4))
    plt.plot(times, err_omega, marker="o", markersize=3, label="L2 error in Omega")
    plt.plot(times, err_gamma, marker="o", markersize=3, label="L2 error on Gamma")
    plt.plot(times, err_total, marker="o", markersize=3, label="combined error")
    #plt.plot(times, rel_total, marker="o", markersize=3, label="relative combined error")
    plt.title("Error history")
    plt.xlabel("time")
    plt.ylabel("error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


# ============================================================
# Main simulation
# ============================================================


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if args.ny % 2 != 0 and np.isclose(args.y_line, 0.5):
        raise ValueError("For y_line=0.5, use even ny, e.g. --ny 80 or --ny 96")

    if rank == 0:
        os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------
    # Mesh and function space
    # ------------------------------------------------------------

    domain = mesh.create_unit_square(
        comm,
        args.nx,
        args.ny,
        cell_type=mesh.CellType.triangle
    )

    V = fem.functionspace(domain, ("Lagrange", 1))

    # ------------------------------------------------------------
    # 1D inclusion Gamma
    # ------------------------------------------------------------

    facet_tags, inclusion_facets = create_internal_line_tags(domain, args.y_line)
    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain)
    dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags)

    if rank == 0:
        print("FEniCSx FEM model: 2D reaction-diffusion + 1D internal inclusion")
        print(f"mode = {args.mode}")
        print(f"mesh = {args.nx} x {args.ny}")
        print(f"Gamma = {{y = {args.y_line}}}")
        print(f"marked inclusion facets = {len(inclusion_facets)}")
        print(f"T = {args.T}, steps = {args.steps}")
        print(f"D_omega = {args.D_omega}, c_omega = {args.c_omega}")
        print(f"beta_boundary = {args.beta_boundary}")
        print(f"alpha_gamma = {args.alpha_gamma}, D_gamma = {args.D_gamma}, c_gamma = {args.c_gamma}")

    # ------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------
    # In manufactured mode we impose u=0 on boundary, since the chosen exact solution is zero there.
    # In demo mode no Dirichlet boundary condition is imposed.
    # If beta_boundary=0, the natural zero-flux boundary appears.
    # If beta_boundary>0, a Robin leakage term is added later to the weak form:
    #     D_omega * grad(u) · n + beta_boundary * u = 0 on boundary.

    bcs = []
    if args.mode == "manufactured":
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain,
            fdim,
            lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
        bcs = [bc]
    # ------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------

    u_n = fem.Function(V)
    u_h = fem.Function(V)
    u_exact_final = fem.Function(V)
    error_function = fem.Function(V)

    if args.mode == "manufactured":
        u_n.interpolate(lambda x: exact_np(x, 0.0, args.shift_y))
    else:
        def demo_initial(x):
            return np.exp(-((x[0] - args.demo_x0) ** 2 + (x[1] - args.demo_y0) ** 2) / (2.0 * args.demo_sigma ** 2))
        u_n.interpolate(demo_initial)

    u_n.x.scatter_forward()

    # ------------------------------------------------------------
    # UFL expressions and constants
    # ------------------------------------------------------------

    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    dt = fem.Constant(domain, PETSc.ScalarType(args.T / args.steps))

    D_omega = fem.Constant(domain, PETSc.ScalarType(args.D_omega))
    c_omega = fem.Constant(domain, PETSc.ScalarType(args.c_omega))
    beta_boundary = fem.Constant(domain, PETSc.ScalarType(args.beta_boundary))

    alpha_gamma = fem.Constant(domain, PETSc.ScalarType(args.alpha_gamma))
    D_gamma = fem.Constant(domain, PETSc.ScalarType(args.D_gamma))
    c_gamma = fem.Constant(domain, PETSc.ScalarType(args.c_gamma))

    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # ------------------------------------------------------------
    # Source terms
    # ------------------------------------------------------------

    if args.mode == "manufactured":
        u_ex = exact_ufl(x, t, args.shift_y)
        u_t = -u_ex

        # Volume source:
        # u_t - D_omega * div(grad(u)) + c_omega*u = f_omega
        f_omega = u_t - D_omega * ufl.div(ufl.grad(u_ex)) + c_omega * u_ex

        # Line source on Gamma:
        # alpha*u_t - D_gamma*u_ss + c_gamma*u = f_gamma
        # For horizontal Gamma, s = x, so u_s = du/dx and u_ss = d2u/dx2.
        u_s = ufl.grad(u_ex)[0]
        u_ss = ufl.grad(u_s)[0]
        f_gamma = alpha_gamma * u_t - D_gamma * u_ss + c_gamma * u_ex
    else:
        f_omega = fem.Constant(domain, PETSc.ScalarType(0.0))
        f_gamma = fem.Constant(domain, PETSc.ScalarType(0.0))

    # ------------------------------------------------------------
    # Variational formulation, Backward Euler
    # ------------------------------------------------------------
    # Omega terms:
    # ∫_Omega u^{n+1} v dx
    # + dt ∫_Omega D grad(u^{n+1})·grad(v) dx
    # + dt ∫_Omega c u^{n+1} v dx
    #
    # Gamma terms:
    # + alpha ∫_Gamma u^{n+1} v ds
    # + dt ∫_Gamma D_gamma u_s^{n+1} v_s ds
    # + dt ∫_Gamma c_gamma u^{n+1} v ds
    #
    # Robin boundary leakage in demo mode, if beta_boundary > 0:
    # + dt ∫_boundary beta_boundary * u^{n+1} v ds
    # This corresponds to D_omega * grad(u)·n + beta_boundary*u = 0.
    #
    # RHS:
    # ∫_Omega u^n v dx + dt ∫_Omega f v dx
    # + alpha ∫_Gamma u^n v ds + dt ∫_Gamma f_gamma v ds

    a_omega = (
        u_trial * v * dx
        + dt * D_omega * ufl.dot(ufl.grad(u_trial), ufl.grad(v)) * dx
        + dt * c_omega * u_trial * v * dx
    )

    L_omega = u_n * v * dx + dt * f_omega * v * dx

    a_gamma = (
        alpha_gamma * ufl.avg(u_trial) * ufl.avg(v) * dS(1)
        + dt * D_gamma * ufl.avg(ufl.grad(u_trial)[0]) * ufl.avg(ufl.grad(v)[0]) * dS(1)
        + dt * c_gamma * ufl.avg(u_trial) * ufl.avg(v) * dS(1)
    )

    L_gamma = (
        alpha_gamma * ufl.avg(u_n) * ufl.avg(v) * dS(1)
        + dt * ufl.avg(f_gamma) * ufl.avg(v) * dS(1)
    )

    a = a_omega + a_gamma
    L = L_omega + L_gamma

    # Optional Robin boundary condition in demo mode:
    # D_omega * grad(u)·n + beta_boundary*u = 0 on the external boundary.
    # beta_boundary=0 gives the previous natural zero-flux boundary.
    if args.mode == "demo" and args.beta_boundary > 0.0:
        a += dt * beta_boundary * u_trial * v * ds

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        u=u_h,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )

    # ------------------------------------------------------------
    # Error forms for manufactured mode
    # ------------------------------------------------------------

    if args.mode == "manufactured":
        e = u_h - u_ex
        err_omega_form = fem.form(e * e * dx)
        norm_omega_form = fem.form(u_ex * u_ex * dx)

        err_gamma_form = fem.form(ufl.avg(e) * ufl.avg(e) * dS(1))
        norm_gamma_form = fem.form(ufl.avg(u_ex) * ufl.avg(u_ex) * dS(1))
    else:
        err_omega_form = None
        norm_omega_form = None
        err_gamma_form = None
        norm_gamma_form = None

    # ------------------------------------------------------------
    # Initial plots
    # ------------------------------------------------------------

    if rank == 0:
        save_field_plot(
            V, u_n,
            os.path.join(args.output, "initial_condition.png"),
            "Initial condition",
            y_line=args.y_line
        )
        save_vertical_profile(
            V, u_n,
            os.path.join(args.output, "initial_vertical_profile_x05.png"),
            0.5,
            "Initial vertical profile at x=0.5"
        )
        save_line_profile(
            V, u_n, u_exact_final,
            os.path.join(args.output, "initial_trace_on_gamma.png"),
            args.y_line, 0.0, args.shift_y,
            include_exact=(args.mode == "manufactured")
        )

    # ------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------

    times = []
    err_omega_values = []
    err_gamma_values = []
    err_total_values = []
    rel_total_values = []

    save_every = max(1, args.steps // 5)

    for step in range(1, args.steps + 1):
        current_time = step * args.T / args.steps
        t.value = PETSc.ScalarType(current_time)

        problem.solve()
        u_h.x.scatter_forward()

        if args.mode == "manufactured":
            local_err_omega_sq = fem.assemble_scalar(err_omega_form)
            local_norm_omega_sq = fem.assemble_scalar(norm_omega_form)
            local_err_gamma_sq = fem.assemble_scalar(err_gamma_form)
            local_norm_gamma_sq = fem.assemble_scalar(norm_gamma_form)

            err_omega_sq = comm.allreduce(local_err_omega_sq, op=MPI.SUM)
            norm_omega_sq = comm.allreduce(local_norm_omega_sq, op=MPI.SUM)
            err_gamma_sq = comm.allreduce(local_err_gamma_sq, op=MPI.SUM)
            norm_gamma_sq = comm.allreduce(local_norm_gamma_sq, op=MPI.SUM)

            err_omega = np.sqrt(err_omega_sq)
            err_gamma = np.sqrt(err_gamma_sq)
            err_total = np.sqrt(err_omega_sq + err_gamma_sq)
            norm_total = np.sqrt(norm_omega_sq + norm_gamma_sq)
            rel_total = err_total / norm_total if norm_total > 0 else np.nan

            times.append(current_time)
            err_omega_values.append(err_omega)
            err_gamma_values.append(err_gamma)
            err_total_values.append(err_total)
            rel_total_values.append(rel_total)

            if rank == 0:
                print(
                    f"step {step:04d}/{args.steps}, "
                    f"t={current_time:.5f}, "
                    f"err_Omega={err_omega:.6e}, "
                    f"err_Gamma={err_gamma:.6e}, "
                    f"combined={err_total:.6e}, "
                    f"rel={rel_total:.6e}"
                )
        else:
            if rank == 0:
                max_u = np.max(u_h.x.array.real)
                print(f"step {step:04d}/{args.steps}, t={current_time:.5f}, max_u={max_u:.6e}")

        if rank == 0 and (step % save_every == 0 or step == args.steps):
            save_field_plot(
                V, u_h,
                os.path.join(args.output, f"solution_step_{step:04d}.png"),
                f"Solution u(x,y,t), t={current_time:.3f}",
                y_line=args.y_line
            )
            save_vertical_profile(
                V, u_h,
                os.path.join(args.output, f"vertical_profile_x05_step_{step:04d}.png"),
                0.5,
                f"Vertical profile at x=0.5, t={current_time:.3f}"
            )
            save_line_profile(
                V, u_h, u_exact_final,
                os.path.join(args.output, f"trace_on_gamma_step_{step:04d}.png"),
                args.y_line, current_time, args.shift_y,
                include_exact=(args.mode == "manufactured")
            )

        u_n.x.array[:] = u_h.x.array
        u_n.x.scatter_forward()

    # ------------------------------------------------------------
    # Final outputs
    # ------------------------------------------------------------

    if rank == 0:
        save_field_plot(
            V, u_h,
            os.path.join(args.output, "final_solution.png"),
            "Final numerical solution",
            y_line=args.y_line
        )

        save_vertical_profile(
            V, u_h,
            os.path.join(args.output, "final_vertical_profile_x05.png"),
            0.5,
            "Final vertical profile at x=0.5"
        )

        save_line_profile(
            V, u_h, u_exact_final,
            os.path.join(args.output, "final_trace_on_gamma.png"),
            args.y_line, args.T, args.shift_y,
            include_exact=(args.mode == "manufactured")
        )

        if args.mode == "manufactured":
            u_exact_final.interpolate(lambda x: exact_np(x, args.T, args.shift_y))
            u_exact_final.x.scatter_forward()
            error_function.x.array[:] = u_h.x.array - u_exact_final.x.array
            error_function.x.scatter_forward()

            save_field_plot(
                V, u_exact_final,
                os.path.join(args.output, "final_exact_solution.png"),
                "Final exact solution",
                y_line=args.y_line
            )
            save_field_plot(
                V, error_function,
                os.path.join(args.output, "final_error_field.png"),
                "Final error field",
                y_line=args.y_line
            )
            save_error_history(
                times,
                err_omega_values,
                err_gamma_values,
                err_total_values,
                rel_total_values,
                os.path.join(args.output, "error_history.png")
            )

            with open(os.path.join(args.output, "errors.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "L2_error_Omega", "L2_error_Gamma", "combined_error", "relative_combined_error"])
                for row in zip(times, err_omega_values, err_gamma_values, err_total_values, rel_total_values):
                    writer.writerow(row)

        with open(os.path.join(args.output, "run_info.txt"), "w", encoding="utf-8") as f:
            f.write("FEniCSx FEM model: 2D reaction-diffusion + 1D internal inclusion\n")
            f.write(f"mode = {args.mode}\n")
            f.write(f"Omega = [0,1] x [0,1]\n")
            f.write(f"Gamma = {{(x,y): y = {args.y_line}}}\n")
            f.write(f"nx = {args.nx}, ny = {args.ny}\n")
            f.write(f"T = {args.T}, steps = {args.steps}\n")
            f.write(f"D_omega = {args.D_omega}\n")
            f.write(f"c_omega = {args.c_omega}\n")
            f.write(f"beta_boundary = {args.beta_boundary}\n")
            f.write(f"alpha_gamma = {args.alpha_gamma}\n")
            f.write(f"D_gamma = {args.D_gamma}\n")
            f.write(f"c_gamma = {args.c_gamma}\n")
            if args.mode == "manufactured" and len(err_total_values) > 0:
                f.write("\nExact solution:\n")
                f.write("u_exact = exp(-t) sin(pi*x) sin(pi*y) (1 + shift_y*(y-0.5))\n")
                f.write(f"shift_y = {args.shift_y}\n")
                f.write("\nFinal errors:\n")
                f.write(f"L2_error_Omega = {err_omega_values[-1]:.12e}\n")
                f.write(f"L2_error_Gamma = {err_gamma_values[-1]:.12e}\n")
                f.write(f"combined_error = {err_total_values[-1]:.12e}\n")
                f.write(f"relative_combined_error = {rel_total_values[-1]:.12e}\n")
            else:
                if args.beta_boundary > 0.0:
                    f.write("\nDemo mode: Robin boundary leakage condition is used; no exact error is computed.\n")
                    f.write("Boundary condition: D_omega * grad(u)·n + beta_boundary*u = 0 on boundary.\n")
                else:
                    f.write("\nDemo mode: natural zero-flux boundary conditions; no exact error is computed.\n")

        print(f"Simulation completed. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
