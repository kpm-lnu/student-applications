import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from src.assembly import assemble_mass_matrix, assemble_stiffness_matrix
from src.mesh import TriangularMesh
from src.solver import Solver

D = 0.01
T = 0.5
dt = 0.005
nx = ny = 32

u0_func = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
u_exact_func = lambda x, y, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2 * np.pi**2 * D * t)


mesh_ours = TriangularMesh(1.0, 1.0, nx, ny)
M_ours = assemble_mass_matrix(mesh_ours)
K_ours = assemble_stiffness_matrix(mesh_ours, D)
C_zero = csc_matrix((mesh_ours.n_nodes, mesh_ours.n_nodes))

u0_ours = np.array([u0_func(x, y) for x, y in mesh_ours.nodes])
solver_ours = Solver(mesh_ours, M_ours, K_ours, C_zero, dt, theta=1.0)
sol_ours, _ = solver_ours.solve(
    u0_ours, T, g_func=lambda x, y, t: 0.0, store_every=10000
)
u_final_ours = sol_ours[-1]


from skfem import Basis, ElementTriP1, MeshTri
from skfem.models.poisson import laplace, mass

skfem_mesh = MeshTri(mesh_ours.nodes.T, mesh_ours.elements.T)
basis = Basis(skfem_mesh, ElementTriP1())

M_sk = mass.assemble(basis)
K_sk = D * laplace.assemble(basis)

u_sk = u0_func(*skfem_mesh.p)
boundary_nodes = skfem_mesh.boundary_nodes()
n_steps = int(np.ceil(T / dt))

for _ in range(n_steps):
    A = M_sk + dt * K_sk
    b = M_sk @ u_sk

    A_mod = A.tolil()
    for node in boundary_nodes:
        A_mod[node, :] = 0
        A_mod[node, node] = 1.0
        b[node] = 0.0
    A_mod = A_mod.tocsc()

    u_sk = spsolve(A_mod, b)

u_final_sk = u_sk


u_exact = np.array([u_exact_func(x, y, T) for x, y in mesh_ours.nodes])

diff_ours_exact = np.abs(u_final_ours - u_exact)
diff_sk_exact = np.abs(u_final_sk - u_exact)
diff_ours_sk = np.abs(u_final_ours - u_final_sk)

print("=" * 55)
print("  Порівняння: in-house солвер vs scikit-fem vs точний")
print("=" * 55)
print(f"  Сітка: {nx}x{ny}, вузлів: {mesh_ours.n_nodes}")
print(f"  D = {D}, T = {T}, dt = {dt}")
print()
print(f"  |наш - точний|       max = {diff_ours_exact.max():.6e}   mean = {diff_ours_exact.mean():.6e}")
print(f"  |skfem - точний|     max = {diff_sk_exact.max():.6e}   mean = {diff_sk_exact.mean():.6e}")
print(f"  |наш - skfem|        max = {diff_ours_sk.max():.6e}   mean = {diff_ours_sk.mean():.6e}")
print()

rel = diff_ours_sk.max() / max(np.abs(u_exact).max(), 1e-16)
print(f"  Відносна різниця ours vs skfem: {rel:.2e}")


triang = mtri.Triangulation(mesh_ours.nodes[:, 0], mesh_ours.nodes[:, 1],
                            mesh_ours.elements)

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

axes[0].tricontourf(triang, u_final_ours, levels=15, cmap='viridis')
axes[0].set_title('In-house солвер', fontsize=11)
axes[0].set_aspect('equal'); axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

axes[1].tricontourf(triang, u_final_sk, levels=15, cmap='viridis')
axes[1].set_title('scikit-fem', fontsize=11)
axes[1].set_aspect('equal'); axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

axes[2].tricontourf(triang, u_exact, levels=15, cmap='viridis')
axes[2].set_title('Точний розв\'язок', fontsize=11)
axes[2].set_aspect('equal'); axes[2].set_xlabel('x'); axes[2].set_ylabel('y')

tcf = axes[3].tricontourf(triang, diff_ours_sk, levels=15, cmap='hot_r')
fig.colorbar(tcf, ax=axes[3], shrink=0.85)
axes[3].set_title(f'|in-house - skfem|\nmax = {diff_ours_sk.max():.2e}', fontsize=10)
axes[3].set_aspect('equal'); axes[3].set_xlabel('x'); axes[3].set_ylabel('y')

fig.suptitle(f'Верифікація проти scikit-fem (сітка {nx}x{ny}, t={T})',
             fontsize=13, y=1.02)
fig.savefig("results/verification_vs_skfem.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("  -> results/verification_vs_skfem.png")
