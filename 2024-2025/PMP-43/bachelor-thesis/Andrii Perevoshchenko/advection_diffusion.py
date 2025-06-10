from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from ufl import interval
from fenics import avg, dS, CellDiameter
import argparse
import os
from datetime import datetime
import pandas as pd
from scipy.interpolate import interp1d


def main():
    parser = argparse.ArgumentParser(description="1D Advection-Diffusion Example Selector")
    parser.add_argument('--example', choices=['example1', 'example2', 'example3'], default='example1',
                        help='Choose example case')
    parser.add_argument('--degree', type=int, choices=[1,2,3], default=2, help='Polynomial degree of finite element (1, 2, or 3)')
    parser.add_argument('--stabilization', choices=['none', 'artificial', 'streamline', 'bubble'], default='none',
                        help='Stabilization method: none, artificial, streamline diffusion, or bubble functions')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Artificial diffusion parameter (default: 1.0)')
    parser.add_argument('--delta', type=float, default=0.5,
                        help='Streamline diffusion parameter (default: 0.5)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.example}_{timestamp}_{args.stabilization}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in directory: {output_dir}")

    def get_peclet_regimes(L, nx):
        h = L/nx  
        return [
            {'name': 'diffusion', 'D': 1.0, 'velocity': 0.1*h},    
            {'name': 'mixed', 'D': 0.1, 'velocity': 0.1},          
            {'name': 'transition', 'D': 1e-3, 'velocity': 1.0},    
            {'name': 'advection', 'D': 1e-9, 'velocity': 1.0}     
        ]

    # Example parameter sets
    if args.example == 'example1':
        L = 1.0
        nx = 200
        T = 1.0
        u_0 = Expression('sin(pi*x[0])', degree=2)
        f = Constant(0.0)  
        bc_type_left = 'Dirichlet'
        bc_type_right = 'Dirichlet'
        bc_left = Constant(0.0)
        bc_right = Constant(0.0)
        print('Example 1: Dirichlet BC')
    elif args.example == 'example2':
        L = 1.0
        nx = 200
        T = 1.0
        u_0 = Expression('exp(-(x[0]-0.2)*(x[0]-0.2)/0.01)', degree=2)
        f = Expression('(2*D/0.01)*(1 - (x[0]-0.2)*(x[0]-0.2)/0.01)*exp(-(x[0]-0.2)*(x[0]-0.2)/0.01) - (2*alpha*(x[0]-0.2)/0.01)*exp(-(x[0]-0.2)*(x[0]-0.2)/0.01)',
                      D=1.0, alpha=1.0, degree=2)  
        bc_type_left = 'Dirichlet'
        bc_type_right = 'Dirichlet'
        bc_left = Constant(1.0)
        bc_right = Constant(0.0)
        print('Example 2: Modified Dirichlet BC')
    elif args.example == 'example3':
        L = 2.0
        nx = 200
        T = 2.0
        u_0 = Expression('exp(-(x[0]-0.5)*(x[0]-0.5)/0.02)', degree=2)
        f = Expression('(2*D/0.02)*(1 - (x[0]-0.5)*(x[0]-0.5)/0.02)*exp(-(x[0]-0.5)*(x[0]-0.5)/0.02) - (2*alpha*(x[0]-0.5)/0.02)*exp(-(x[0]-0.5)*(x[0]-0.5)/0.02)',
                      D=1.0, alpha=1.0, degree=2) 
        bc_type_left = 'Robin'
        bc_type_right = 'Dirichlet'
        bc_left = (1.0, 5.0)
        bc_right = Constant(0.0)
        print('Example 3: Mixed BC with different domain')
    else:
        D = 0.00001
        velocity = 1.0
        L = 1.0
        nx = 100
        T = 5.0
        u_0 = Expression('exp(-(x[0]-0.2)*(x[0]-0.2)/0.01)', degree=2)
        bc_type_left = 'Neumann'
        bc_type_right = 'Dirichlet'
        bc_left = Constant(0.0)
        bc_right = Constant(0.0)
        print('High Peclet number: advection-dominated')

    num_steps = 500
    dt = T / num_steps

    nx_list = [32, 64, 128, 256, 512]
    regimes = ['diffusion', 'mixed', 'advection', 'transition']
    for regime_name in regimes:
        error_results = []
        reference_solutions = {}
        nx_ref = 512
        peclet_regimes = get_peclet_regimes(L, nx_ref)
        regime = next(r for r in peclet_regimes if r['name'] == regime_name)
        D = regime['D']
        velocity = regime['velocity']
        h = L/nx_ref
        Pe_local = abs(velocity) * h / (2 * D)
        Pe_global = velocity * L / D
        mesh = IntervalMesh(nx_ref, 0, L)
        Pk = FiniteElement('P', interval, args.degree)
        elem = Pk
        V = FunctionSpace(mesh, elem)
        u_n = project(u_0, V)
        def boundary_left(x, on_boundary):
            return on_boundary and near(x[0], 0)
        def boundary_right(x, on_boundary):
            return on_boundary and near(x[0], L)
        bcs = []
        robin_left = False
        robin_right = False
        if bc_type_left == 'Dirichlet':
            bcs.append(DirichletBC(V, bc_left, boundary_left))
        elif bc_type_left == 'Robin':
            robin_left = True
        if bc_type_right == 'Dirichlet':
            bcs.append(DirichletBC(V, bc_right, boundary_right))
        elif bc_type_right == 'Robin':
            robin_right = True
        boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        class Left(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0) and on_boundary
        class Right(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], L) and on_boundary
        Left().mark(boundaries, 1)
        Right().mark(boundaries, 2)
        ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
        u = TrialFunction(V)
        v = TestFunction(V)
        F = (u - u_n)/dt * v * dx + velocity * u.dx(0) * v * dx + D * u.dx(0) * v.dx(0) * dx - f * v * dx
        if args.stabilization == 'artificial':
            D_art = args.alpha * abs(velocity) * h / 2.0
            F += D_art * u.dx(0) * v.dx(0) * dx
        elif args.stabilization == 'streamline':
            delta = args.delta
            F += delta * h * velocity * u.dx(0) * (v + velocity * v.dx(0)) * dx
        elif args.stabilization == 'bubble':
            Pe = abs(velocity) * h / (2 * D)
            if Pe != 0:
                tau = h / (2 * abs(velocity)) * (1/np.tanh(Pe) - 1/Pe)
            else:
                tau = h*h/(12*D)
            F += tau * (u - u_n)/dt * v.dx(0) * dx
            F += tau * D * u.dx(0) * v.dx(0) * dx
            F += tau * velocity * u.dx(0) * v.dx(0) * dx
            f = Constant(0.0)
            F += tau * f * v.dx(0) * dx
        if robin_left:
            aL, bL = bc_left
            F += aL * u * v * ds(1) + bL * u.dx(0) * v * ds(1)
        if robin_right:
            aR, bR = bc_right
            F += aR * u * v * ds(2) + bR * u.dx(0) * v * ds(2)
        a, L_form = lhs(F), rhs(F)
        u_sol = Function(V)
        t = 0
        times = np.linspace(0, T, num_steps+1)
        solutions = []
        solutions.append(project(u_n, V).vector().get_local())
        for n in range(num_steps):
            t += dt
            solve(a == L_form, u_sol, bcs)
            solutions.append(project(u_sol, V).vector().get_local())
            u_n.assign(u_sol)
        Pk_plot = FunctionSpace(mesh, 'P', args.degree)
        for i in np.linspace(0, len(solutions)-1, 8, dtype=int):
            u_plot = Function(V)
            u_plot.vector()[:] = solutions[i]
            u_proj = project(u_plot, Pk_plot)
            reference_solutions[i] = u_proj.compute_vertex_values(mesh)
            
        plt.figure(figsize=(10, 6))
        x = np.linspace(0, L, nx_ref+1)
        for i in np.linspace(0, len(solutions)-1, 8, dtype=int):
            u_plot = Function(V)
            u_plot.vector()[:] = solutions[i]
            u_proj = project(u_plot, Pk_plot)
            plt.plot(x, u_proj.compute_vertex_values(mesh), label=f't={times[i]:.2f}')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title(f'Reference solution ({regime_name} regime, nx=512)\nPe_global={Pe_global:.2e}, Pe_local={Pe_local:.2e}, stabilization={args.stabilization}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'reference_{regime_name}.png'))
        plt.close()
        for nx in [32, 64, 128, 256]:
            peclet_regimes = get_peclet_regimes(L, nx)
            regime = next(r for r in peclet_regimes if r['name'] == regime_name)
            D = regime['D']
            velocity = regime['velocity']
            h = L/nx
            Pe_local = abs(velocity) * h / (2 * D)
            Pe_global = velocity * L / D
            mesh = IntervalMesh(nx, 0, L)
            Pk = FiniteElement('P', interval, args.degree)
            elem = Pk
            V = FunctionSpace(mesh, elem)
            u_n = project(u_0, V)
            def boundary_left(x, on_boundary):
                return on_boundary and near(x[0], 0)
            def boundary_right(x, on_boundary):
                return on_boundary and near(x[0], L)
            bcs = []
            robin_left = False
            robin_right = False
            if bc_type_left == 'Dirichlet':
                bcs.append(DirichletBC(V, bc_left, boundary_left))
            elif bc_type_left == 'Robin':
                robin_left = True
            if bc_type_right == 'Dirichlet':
                bcs.append(DirichletBC(V, bc_right, boundary_right))
            elif bc_type_right == 'Robin':
                robin_right = True
            boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
            class Left(SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0], 0) and on_boundary
            class Right(SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0], L) and on_boundary
            Left().mark(boundaries, 1)
            Right().mark(boundaries, 2)
            ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
            u = TrialFunction(V)
            v = TestFunction(V)
            F = (u - u_n)/dt * v * dx + velocity * u.dx(0) * v * dx + D * u.dx(0) * v.dx(0) * dx - f * v * dx
            if args.stabilization == 'artificial':
                D_art = args.alpha * abs(velocity) * h / 2.0
                F += D_art * u.dx(0) * v.dx(0) * dx
            elif args.stabilization == 'streamline':
                delta = args.delta
                F += delta * h * velocity * u.dx(0) * (v + velocity * v.dx(0)) * dx
            elif args.stabilization == 'bubble':
                Pe = abs(velocity) * h / (2 * D)
                if Pe != 0:
                    tau = h / (2 * abs(velocity)) * (1/np.tanh(Pe) - 1/Pe)
                else:
                    tau = h*h/(12*D)
                F += tau * (u - u_n)/dt * v.dx(0) * dx
                F += tau * D * u.dx(0) * v.dx(0) * dx
                F += tau * velocity * u.dx(0) * v.dx(0) * dx
                f = Constant(0.0)
                F += tau * f * v.dx(0) * dx
            if robin_left:
                aL, bL = bc_left
                F += aL * u * v * ds(1) + bL * u.dx(0) * v * ds(1)
            if robin_right:
                aR, bR = bc_right
                F += aR * u * v * ds(2) + bR * u.dx(0) * v * ds(2)
            a, L_form = lhs(F), rhs(F)
            u_sol = Function(V)
            t = 0
            times = np.linspace(0, T, num_steps+1)
            solutions = []
            solutions.append(project(u_n, V).vector().get_local())
            for n in range(num_steps):
                t += dt
                solve(a == L_form, u_sol, bcs)
                solutions.append(project(u_sol, V).vector().get_local())
                u_n.assign(u_sol)
            Pk_plot = FunctionSpace(mesh, 'P', args.degree)
            for i in np.linspace(0, len(solutions)-1, 8, dtype=int):
                u_plot = Function(V)
                u_plot.vector()[:] = solutions[i]
                u_proj = project(u_plot, Pk_plot)
                ref_x = np.linspace(0, L, 512+1)
                coarse_x = np.linspace(0, L, nx+1)
                interp_func = interp1d(coarse_x, u_proj.compute_vertex_values(mesh), kind='linear', fill_value='extrapolate')
                u_interp = interp_func(ref_x)
                u_ref = reference_solutions.get(i, None)
                if u_ref is not None:
                    l2_error = np.sqrt(np.sum((u_interp - u_ref)**2) / len(u_ref))
                    error_results.append({
                        'example': args.example,
                        'regime': regime_name,
                        'stabilization': args.stabilization,
                        'nx': nx,
                        'time_idx': i,
                        'l2_error': l2_error
                    })
            results_df = pd.DataFrame()
            results_df['x'] = np.linspace(0, L, nx+1)
            for i in np.linspace(0, len(solutions)-1, 8, dtype=int):
                u_plot = Function(V)
                u_plot.vector()[:] = solutions[i]
                u_proj = project(u_plot, Pk_plot)
                results_df[f't_{times[i]:.2f}'] = u_proj.compute_vertex_values(mesh)
            results_df.to_csv(os.path.join(output_dir, f'{regime_name}_nx_{nx}.csv'), index=False)
            plt.figure(figsize=(10, 6))
            x = np.linspace(0, L, nx+1)
            for i in np.linspace(0, len(solutions)-1, 8, dtype=int):
                u_plot = Function(V)
                u_plot.vector()[:] = solutions[i]
                u_proj = project(u_plot, Pk_plot)
                plt.plot(x, u_proj.compute_vertex_values(mesh), label=f't={times[i]:.2f}')
            plt.xlabel('x')
            plt.ylabel('u')
            plt.title(f'Solution at different times ({regime_name} regime, nx={nx})\nPe_global={Pe_global:.2e}, Pe_local={Pe_local:.2e}, stabilization={args.stabilization}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{regime_name}_nx_{nx}.png'))
            plt.close()
        if len(error_results) > 0:
            errors_df = pd.DataFrame(error_results)
            errors_df = errors_df.sort_values('nx')
            errors_df['order'] = float('nan')
            unique_nx = sorted(errors_df['nx'].unique())
            for t_idx in sorted(errors_df['time_idx'].unique()):
                mask = errors_df['time_idx'] == t_idx
                nx_vals = errors_df[mask]['nx'].values
                err_vals = errors_df[mask]['l2_error'].values
                for i in range(1, len(nx_vals)):
                    h1 = 1.0 / nx_vals[i-1]
                    h2 = 1.0 / nx_vals[i]
                    e1 = err_vals[i-1]
                    e2 = err_vals[i]
                    if e2 > 0 and e1 > 0:
                        order = np.log(e1/e2) / np.log(h1/h2)
                        errors_df.loc[(errors_df['nx'] == nx_vals[i]) & (errors_df['time_idx'] == t_idx), 'order'] = order
            errors_df.to_csv(os.path.join(output_dir, f'convergence_{regime_name}.csv'), index=False)
            plt.figure()
            for t_idx in sorted(errors_df['time_idx'].unique()):
                dft = errors_df[errors_df['time_idx']==t_idx]
                plt.loglog(dft['nx'], dft['l2_error'], marker='o', label=f't_idx={t_idx}')
            plt.xlabel('nx')
            plt.ylabel('L2 error')
            plt.title(f'Convergence: {args.example}, {regime_name}, {args.stabilization}')
            plt.legend()
            plt.grid(True, which='both')
            plt.savefig(os.path.join(output_dir, f'convergence_{regime_name}.png'))
            plt.close()

if __name__ == "__main__":
    main() 