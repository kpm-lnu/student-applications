import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from random import random, randint, seed, uniform
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Параметри сітки
Nx, Ny, Nz = 5, 5, 5
Lx, Ly, Lz = 4.0, 4.0, 4.0
dx, dy, dz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
mu = 1.0
N_nodes = Nx * Ny * Nz

# Індексація в sol_vector
def idx_u_x(i, j, k): return i + Nx*(j + Ny*k)
def idx_u_y(i, j, k): return N_nodes + i + Nx*(j + Ny*k)
def idx_u_z(i, j, k): return 2*N_nodes + i + Nx*(j + Ny*k)
def idx_p(i, j, k):   return 3*N_nodes + i + Nx*(j + Ny*k)

C = 2.5
P_out = 0.0

def residuals_Stokes(sol):
    u_x = np.zeros((Nx, Ny, Nz))
    u_y = np.zeros((Nx, Ny, Nz))
    u_z = np.zeros((Nx, Ny, Nz))
    p   = np.zeros((Nx, Ny, Nz))
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                u_x[i,j,k] = sol[idx_u_x(i,j,k)]
                u_y[i,j,k] = sol[idx_u_y(i,j,k)]
                u_z[i,j,k] = sol[idx_u_z(i,j,k)]
                p[i,j,k]   = sol[idx_p(i,j,k)]
    res = []
    for k in range(1, Nz-1):
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                dudx = (u_x[i+1,j,k]-u_x[i-1,j,k])/(2*dx)
                dudy = (u_y[i,j+1,k]-u_y[i,j-1,k])/(2*dy)
                dudz = (u_z[i,j,k+1]-u_z[i,j,k-1])/(2*dz)
                res.append(dudx + dudy + dudz)
                dpdx = (p[i+1,j,k]-p[i-1,j,k])/(2*dx)
                d2u = ((u_x[i+1,j,k]-2*u_x[i,j,k]+u_x[i-1,j,k])/(dx*dx)
                       +(u_x[i,j+1,k]-2*u_x[i,j,k]+u_x[i,j-1,k])/(dy*dy)
                       +(u_x[i,j,k+1]-2*u_x[i,j,k]+u_x[i,j,k-1])/(dz*dz))
                res.append(dpdx - mu*d2u)
                dpdy = (p[i,j+1,k]-p[i,j-1,k])/(2*dy)
                res.append(dpdy)
                dpdz = (p[i,j,k+1]-p[i,j,k-1])/(2*dz)
                res.append(dpdz)
    i = 0
    for k in range(1, Nz-1):
        for j in range(1, Ny-1):
            yj = j*dy
            ua = C * yj * (4.0 - yj)
            res.append(u_x[i,j,k] - ua)
            res.append(u_y[i,j,k] - 0.0)
            res.append(u_z[i,j,k] - 0.0)
    i = Nx-1
    for k in range(Nz):
        for j in range(Ny):
            res.append(p[i,j,k] - P_out)
    for k in range(Nz):
        for i in range(Nx):
            res.append(u_x[i,0,k] - 0.0)
            res.append(u_y[i,0,k] - 0.0)
            res.append(u_z[i,0,k] - 0.0)
            res.append(u_x[i,Ny-1,k] - 0.0)
            res.append(u_y[i,Ny-1,k] - 0.0)
            res.append(u_z[i,Ny-1,k] - 0.0)
    for j in range(Ny):
        for i in range(Nx):
            res.append(u_x[i,j,0] - 0.0)
            res.append(u_y[i,j,0] - 0.0)
            res.append(u_z[i,j,0] - 0.0)
            res.append(u_x[i,j,Nz-1] - 0.0)
            res.append(u_y[i,j,Nz-1] - 0.0)
            res.append(u_z[i,j,Nz-1] - 0.0)
    return np.array(res)

def enforce_boundaries(sol):
    i = 0
    for k in range(1, Nz-1):
        for j in range(1, Ny-1):
            yj = j*dy
            ua = C * yj * (4.0 - yj)
            sol[idx_u_x(i,j,k)] = ua
            sol[idx_u_y(i,j,k)] = 0.0
            sol[idx_u_z(i,j,k)] = 0.0
    j = 0
    for k in range(Nz):
        for i in range(Nx):
            sol[idx_u_x(i,j,k)] = 0.0
            sol[idx_u_y(i,j,k)] = 0.0
            sol[idx_u_z(i,j,k)] = 0.0
    j = Ny-1
    for k in range(Nz):
        for i in range(Nx):
            sol[idx_u_x(i,j,k)] = 0.0
            sol[idx_u_y(i,j,k)] = 0.0
            sol[idx_u_z(i,j,k)] = 0.0
    k = 0
    for j in range(Ny):
        for i in range(Nx):
            sol[idx_u_x(i,j,k)] = 0.0
            sol[idx_u_y(i,j,k)] = 0.0
            sol[idx_u_z(i,j,k)] = 0.0
    k = Nz-1
    for j in range(Ny):
        for i in range(Nx):
            sol[idx_u_x(i,j,k)] = 0.0
            sol[idx_u_y(i,j,k)] = 0.0
            sol[idx_u_z(i,j,k)] = 0.0
    i = Nx-1
    for k in range(Nz):
        for j in range(Ny):
            sol[idx_p(i,j,k)] = P_out

def fitness(sol):
    r = residuals_Stokes(sol)
    return np.sum(r*r)

pop_size       = 100
generations    = 1000
mutation_rate  = 0.05
crossover_rate = 0.5
elite_fraction = 0.1
tol_improve    = 1e-6
patience       = 50

seed(0)
np.random.seed(0)

population = []
for _ in range(pop_size):
    sol = np.zeros(4*N_nodes)
    for ii in range(4*N_nodes):
        if ii < 3*N_nodes:
            sol[ii] = uniform(0.0, 10.0)
        else:
            sol[ii] = uniform(-1.0, 1.0)
    enforce_boundaries(sol)
    population.append(sol)

fmin_values = []
favg_values = []
fvar_values = []

best_fmin = np.inf
no_improve_count = 0

for gen in range(generations):
    fitness_vals = [fitness(ind) for ind in population]
    fmin = np.min(fitness_vals)
    favg = np.mean(fitness_vals)
    fvar = np.var(fitness_vals)
    fmin_values.append(fmin)
    favg_values.append(favg)
    fvar_values.append(fvar)

    if best_fmin - fmin < tol_improve:
        no_improve_count += 1
    else:
        no_improve_count = 0
        best_fmin = fmin

    if no_improve_count >= patience:
        print(f"Зупинено на поколінні {gen} (стабілізація Fmin: {best_fmin:.3e})")
        break

    sorted_idx = np.argsort(fitness_vals)
    population = [population[i] for i in sorted_idx]

    elite_count = int(elite_fraction * pop_size)
    new_pop = population[:elite_count]

    while len(new_pop) < pop_size:
        p1 = population[randint(0, pop_size//2)]
        p2 = population[randint(0, pop_size//2)]
        child = np.copy(p1)
        if random() < crossover_rate:
            cp = randint(0, len(child)-1)
            child[cp:] = p2[cp:]
        for idx in range(len(child)):
            if random() < mutation_rate:
                if idx < 3*N_nodes:
                    child[idx] += uniform(-0.1, 0.1)*10.0
                else:
                    child[idx] += uniform(-0.5, 0.5)
        enforce_boundaries(child)
        new_pop.append(child)

    population = new_pop

    if gen % 100 == 0:
        print(f"Покоління {gen:4d}:  Fmin = {fmin:.3e},  Favg = {favg:.3e},  Fvar = {fvar:.3e}")

best_sol_GA = population[0]
print(f"\nНайкраще рішення ГА — Fmin: {fitness(best_sol_GA):.3e}")

res_ls = least_squares(
    residuals_Stokes,
    best_sol_GA,
    xtol=1e-12,
    ftol=1e-12,
    gtol=1e-12
)
sol_refined = res_ls.x
print(f"L2-норма нев’язок після least_squares: {np.linalg.norm(residuals_Stokes(sol_refined)):.3e}")

u_x_field = np.zeros((Nx, Ny, Nz))
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            u_x_field[i,j,k] = sol_refined[idx_u_x(i,j,k)]

mid_k = Nz//2
u_x_mid = u_x_field[:,:,mid_k]
X = np.linspace(0, Lx, Nx)
Y = np.linspace(0, Ly, Ny)
Xg, Yg = np.meshgrid(X, Y, indexing='ij')

plt.figure(figsize=(6,5))
cf = plt.contourf(Xg, Yg, u_x_mid, levels=20, cmap='viridis')
plt.colorbar(cf, label='u_x')
plt.title(f"u_x на z={Lz/2:.1f}")
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

eps = 1e-12
fmin_plot = [v if v>0 else eps for v in fmin_values]
favg_plot = [v if v>0 else eps for v in favg_values]
fvar_plot = [v if v>0 else eps for v in fvar_values]

plt.figure(figsize=(8,6))
plt.plot(range(len(fmin_plot)), fmin_plot, label='Fmin', linewidth=2)
plt.plot(range(len(favg_plot)), favg_plot, label='Favg', linewidth=2)
plt.plot(range(len(fvar_plot)), fvar_plot, label='Fvar', linewidth=2)
plt.yscale('log')
plt.xlabel('Покоління')
plt.ylabel('Сума квадратів нев’язок')
plt.title('Динаміка F-метрик (лог-шкала)')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()
