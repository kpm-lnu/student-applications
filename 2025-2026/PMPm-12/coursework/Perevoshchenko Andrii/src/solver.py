import numpy as np
from scipy.sparse.linalg import spsolve

from .boundary import (
    apply_dirichlet_simple,
    assemble_neumann_vector,
    assemble_robin,
    get_dirichlet_nodes_and_values,
)


class Solver:
    def __init__(self, mesh, M, K, C, dt, theta=1.0):
        self.mesh = mesh
        self.M = M
        self.K = K
        self.C = C
        self.dt = dt
        self.theta = theta

        self.A_op = C + K
        self.LHS_base = M + dt * theta * self.A_op
        self.RHS_base = M - dt * (1.0 - theta) * self.A_op

    def solve(self, u0, T, f_func=None, g_func=None, bc_sides=None,
              neumann_func=None, neumann_sides=None,
              robin_alpha=None, robin_beta=None, robin_sides=None,
              store_every=1):
        dt = self.dt
        theta = self.theta
        n_steps = int(np.ceil(T / dt))

        u = u0.copy()
        solutions = [u.copy()]
        times = [0.0]

        if g_func is None:
            g_func = lambda x, y, t: 0.0
        if f_func is None:
            f_func = lambda x, y, t: 0.0

        R_mat = None
        if robin_alpha is not None and robin_sides:
            R_mat, _ = assemble_robin(self.mesh, robin_alpha,
                                      lambda x, y, t: 0.0, 0.0, robin_sides)

        for step in range(1, n_steps + 1):
            t_old = (step - 1) * dt
            t_new = step * dt

            from .assembly import assemble_load_vector
            F_old = assemble_load_vector(self.mesh, f_func, t_old)
            F_new = assemble_load_vector(self.mesh, f_func, t_new)

            rhs = self.RHS_base @ u + dt * (theta * F_new + (1 - theta) * F_old)
            lhs = self.LHS_base.copy()

            if neumann_func is not None and neumann_sides:
                F_N_old = assemble_neumann_vector(
                    self.mesh, neumann_func, t_old, neumann_sides)
                F_N_new = assemble_neumann_vector(
                    self.mesh, neumann_func, t_new, neumann_sides)
                rhs += dt * (theta * F_N_new + (1 - theta) * F_N_old)

            if R_mat is not None:
                lhs = lhs + dt * theta * R_mat
                rhs -= dt * (1 - theta) * R_mat @ u
                _, R_vec_old = assemble_robin(
                    self.mesh, robin_alpha, robin_beta, t_old, robin_sides)
                _, R_vec_new = assemble_robin(
                    self.mesh, robin_alpha, robin_beta, t_new, robin_sides)
                rhs += dt * (theta * R_vec_new + (1 - theta) * R_vec_old)

            bc_nodes, bc_values = get_dirichlet_nodes_and_values(
                self.mesh, g_func, t_new, bc_sides
            )
            lhs, rhs = apply_dirichlet_simple(lhs, rhs, bc_nodes, bc_values)

            u = spsolve(lhs, rhs)

            if step % store_every == 0 or step == n_steps:
                solutions.append(u.copy())
                times.append(t_new)

        return solutions, times

    def solve_stationary(self, f_func=None, g_func=None, bc_sides=None,
                         neumann_func=None, neumann_sides=None,
                         robin_alpha=None, robin_beta=None, robin_sides=None):
        if g_func is None:
            g_func = lambda x, y, t: 0.0
        if f_func is None:
            f_func = lambda x, y, t: 0.0

        from .assembly import assemble_load_vector
        F = assemble_load_vector(self.mesh, f_func, 0.0)

        A = self.A_op.copy()
        b = F.copy()

        if neumann_func is not None and neumann_sides:
            b += assemble_neumann_vector(self.mesh, neumann_func, 0.0,
                                         neumann_sides)

        if robin_alpha is not None and robin_sides:
            R_mat, R_vec = assemble_robin(self.mesh, robin_alpha, robin_beta,
                                          0.0, robin_sides)
            A = A + R_mat
            b += R_vec

        bc_nodes, bc_values = get_dirichlet_nodes_and_values(
            self.mesh, g_func, 0.0, bc_sides
        )
        A, b = apply_dirichlet_simple(A, b, bc_nodes, bc_values)

        return spsolve(A, b)
