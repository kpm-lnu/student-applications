import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Beam and material properties
L = 1.0
H = 0.2
P = 1.0
E = 1.0
nu = 0.3
I = H**3 / 12

# Mesh parameters
Nx, Ny = 80, 20
x = np.linspace(0, L, Nx + 1)
y = np.linspace(-H/2, H/2, Ny + 1)
X, Y = np.meshgrid(x, y)
nodes = np.column_stack((X.ravel(), Y.ravel()))
elements = []

def node_id(i, j): return i * (Nx + 1) + j

for i in range(Ny):
    for j in range(Nx):
        n1 = node_id(i, j)
        n2 = node_id(i, j + 1)
        n3 = node_id(i + 1, j + 1)
        n4 = node_id(i + 1, j)
        elements.append([n1, n2, n3, n4])

# FEM setup
n_dof = nodes.shape[0] * 2
K = lil_matrix((n_dof, n_dof))
F = np.zeros(n_dof)

# Plane stress D matrix
D = (E / (1 - nu**2)) * np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1 - nu)/2]
])

# Shape function derivatives in reference coords
def dN_dxi(xi, eta):
    return 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],
        [ (1 - eta), -(1 + xi)],
        [ (1 + eta),  (1 + xi)],
        [-(1 + eta),  (1 - xi)]
    ]).T

# Gauss points for integration
gauss_pts = [(-1/np.sqrt(3), -1/np.sqrt(3)),
             ( 1/np.sqrt(3), -1/np.sqrt(3)),
             ( 1/np.sqrt(3),  1/np.sqrt(3)),
             (-1/np.sqrt(3),  1/np.sqrt(3))]

# Shape function
def shape_func(xi, eta):
    return 0.25 * np.array([
        (1 - xi)*(1 - eta),
        (1 + xi)*(1 - eta),
        (1 + xi)*(1 + eta),
        (1 - xi)*(1 + eta)
    ])

# Assemble stiffness matrix
for el in elements:
    coords = nodes[el]
    Ke = np.zeros((8, 8))
    for xi, eta in gauss_pts:
        N = shape_func(xi, eta)
        dN_ref = dN_dxi(xi, eta)
        J = dN_ref @ coords
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        dN = invJ @ dN_ref

        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i]     = dN[0, i]
            B[1, 2*i+1]   = dN[1, i]
            B[2, 2*i]     = dN[1, i]
            B[2, 2*i+1]   = dN[0, i]

        Ke += (B.T @ D @ B) * detJ

    dof = np.array([[2*n, 2*n+1] for n in el]).flatten()
    for i in range(8):
        for j in range(8):
            K[dof[i], dof[j]] += Ke[i, j]

# Apply traction P at free (right) edge
for i in range(Ny):
    n1 = node_id(i, Nx)
    n2 = node_id(i + 1, Nx)
    length = np.linalg.norm(nodes[n2] - nodes[n1])
    for n in [n1, n2]:
        F[2*n + 1] += P * length / 2 / H

# Clamp left edge (x = 0)
for i in range(Ny + 1):
    n = node_id(i, 0)
    K[2*n, :] = 0
    K[2*n+1, :] = 0
    K[2*n, 2*n] = 1
    K[2*n+1, 2*n+1] = 1
    F[2*n] = 0
    F[2*n+1] = 0

# Solve system
U = spsolve(K.tocsr(), F).reshape(-1, 2)
ux, uy = U[:, 0], U[:, 1]

# Analytical solution
x_vals = nodes[:, 0]
y_vals = nodes[:, 1]
ux_ana = -P * y_vals * (3*L - x_vals) * x_vals / (6 * E * I)
uy_ana = P * (3*L * x_vals**2 - x_vals**3) / (6 * E * I)

# Plot displacements
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

sc1 = axs[0, 0].tricontourf(x_vals, y_vals, ux, 20)
axs[0, 0].set_title("Numerical $u_x$")
fig.colorbar(sc1, ax=axs[0, 0])

sc2 = axs[0, 1].tricontourf(x_vals, y_vals, ux_ana, 20)
axs[0, 1].set_title("Analytical $u_x$")
fig.colorbar(sc2, ax=axs[0, 1])

sc3 = axs[1, 0].tricontourf(x_vals, y_vals, uy, 20)
axs[1, 0].set_title("Numerical $u_y$")
fig.colorbar(sc3, ax=axs[1, 0])

sc4 = axs[1, 1].tricontourf(x_vals, y_vals, uy_ana, 20)
axs[1, 1].set_title("Analytical $u_y$")
fig.colorbar(sc4, ax=axs[1, 1])

for ax in axs.flat:
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect('equal')
    ax.grid(True)

plt.tight_layout()
plt.show()

# === Compute FEM stress at element centers ===
sigma_xx_fem = []
sigma_yy_fem = []
sigma_xy_fem = []
centers_x = []
centers_y = []

for el in elements:
    coords = nodes[el]
    u_el = U[el].flatten()

    xi, eta = 0, 0  # center of the element
    dN_ref = dN_dxi(xi, eta)
    J = dN_ref @ coords
    invJ = np.linalg.inv(J)
    dN = invJ @ dN_ref

    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2*i]     = dN[0, i]
        B[1, 2*i+1]   = dN[1, i]
        B[2, 2*i]     = dN[1, i]
        B[2, 2*i+1]   = dN[0, i]

    stress = D @ (B @ u_el)
    sigma_xx_fem.append(stress[0])
    sigma_yy_fem.append(stress[1])
    sigma_xy_fem.append(stress[2])
    center = coords.mean(axis=0)
    centers_x.append(center[0])
    centers_y.append(center[1])

# === Analytical stress: only σ_xx is non-zero ===
sigma_xx_ana = -P * np.array(centers_y) * (L - np.array(centers_x)) / I
sigma_yy_ana = np.zeros_like(sigma_xx_ana)
sigma_xy_ana = np.zeros_like(sigma_xx_ana)

# === Plot comparison ===
fig, axs = plt.subplots(3, 2, figsize=(14, 12))

# σ_xx
axs[0, 0].tricontourf(centers_x, centers_y, sigma_xx_fem, 20)
axs[0, 0].set_title("FEM $\\sigma_{xx}$")
axs[0, 1].tricontourf(centers_x, centers_y, sigma_xx_ana, 20)
axs[0, 1].set_title("Analytical $\\sigma_{xx}$")

# σ_yy
axs[1, 0].tricontourf(centers_x, centers_y, sigma_yy_fem, 20)
axs[1, 0].set_title("FEM $\\sigma_{yy}$")
axs[1, 1].tricontourf(centers_x, centers_y, sigma_yy_ana, 20)
axs[1, 1].set_title("Analytical $\\sigma_{yy}$")

# σ_xy
axs[2, 0].tricontourf(centers_x, centers_y, sigma_xy_fem, 20)
axs[2, 0].set_title("FEM $\\sigma_{xy}$")
axs[2, 1].tricontourf(centers_x, centers_y, sigma_xy_ana, 20)
axs[2, 1].set_title("Analytical $\\sigma_{xy}$")

for ax in axs.flat:
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect('equal')
    ax.grid(True)

plt.tight_layout()
plt.show()

