import numpy as np
import matplotlib.pyplot as plt
from mesh_generator import generate_nonmatching_subdomains
from assembler_solver import assemble_global_stiffness, apply_dirichlet_bc, solve_system, B_matrix_Q4
from fem_core import dN_dxi_Q4
from mortar import find_interface_nodes

# ---------------- Beam & Material ----------------
L = 1.0      # length
H = 0.2      # height
P = 1.0      # tip load
E = 1.0      # Young’s modulus
nu = 0.3     # Poisson’s ratio
I = H**3 / 12

# Elasticity matrix (plane stress)
D = (E / (1 - nu**2)) * np.array([
    [1,  nu, 0],
    [nu, 1,  0],
    [0,  0,  (1 - nu)/2]
])

# ---------------- Mesh ----------------
Nx1, Ny1 = 40, 20          # left sub-domain
Nx2, Ny2 = 40, 20          # right sub-domain
nodes1, elems1, nodes2, elems2, _, _ = generate_nonmatching_subdomains(Nx1, Ny1, Nx2, Ny2)

# --- scale & shift y so domain height = H and neutral axis at y=0 ---
for arr in (nodes1, nodes2):
    # original y runs 0 … 1  → scale by H and shift down by H/2
    arr[:,1] = (arr[:,1] - 0.5) * H
nd1, nd2 = len(nodes1), len(nodes2)

# ---------------- Assemble K, F for each part ----------------
K1 = assemble_global_stiffness(nodes1, elems1, D)
K2 = assemble_global_stiffness(nodes2, elems2, D)
F1 = np.zeros(2*nd1)
F2 = np.zeros(2*nd2)

# Apply vertical traction P on right edge of domain‑2
for i, (x, _) in enumerate(nodes2):
    if np.isclose(x, L):
        F2[2*i + 1] += P / Ny2   # lumped per node

# Clamp left edge of domain‑1
fixed = [2*i   for i,(x,_) in enumerate(nodes1) if np.isclose(x,0.0)] + \
        [2*i+1 for i,(x,_) in enumerate(nodes1) if np.isclose(x,0.0)]

# ---------------- Mortar coupling (1‑to‑1) ----------------
slave_ids  = sorted(find_interface_nodes(nodes1), key=lambda idx: nodes1[idx][1])
master_ids = sorted(find_interface_nodes(nodes2), key=lambda idx: nodes2[idx][1])
assert len(slave_ids)==len(master_ids), "Interface node counts differ – simple coupling needs match"
Lagr = len(slave_ids)
rows = 2*Lagr  # ux + uy per node
B1 = np.zeros((rows, 2*nd1))
B2 = np.zeros((rows, 2*nd2))
for k,(s,m) in enumerate(zip(slave_ids, master_ids)):
    B1[2*k,   2*s   ] = 1.0; B2[2*k,   2*m   ] = 1.0  # x
    B1[2*k+1, 2*s+1 ] = 1.0; B2[2*k+1, 2*m+1 ] = 1.0  # y

# ---------------- Saddle‑point system ----------------
K = np.block([
    [K1,                       np.zeros((2*nd1, 2*nd2)),   B1.T],
    [np.zeros((2*nd2, 2*nd1)), K2,                        -B2.T],
    [B1,                      -B2,                         np.zeros((rows, rows))]
])
F = np.concatenate([F1, F2, np.zeros(rows)])
K_bc, F_bc = apply_dirichlet_bc(K, F, fixed)

# ---------------- Solve ----------------
U = solve_system(K_bc, F_bc)
U1 = U[:2*nd1].reshape(-1,2)
U2 = U[2*nd1:2*(nd1+nd2)].reshape(-1,2)
ux_all = np.concatenate([U1[:,0], U2[:,0]])
uy_all = np.concatenate([U1[:,1], U2[:,1]])
all_nodes = np.vstack([nodes1, nodes2])

# ---------------- Analytical displacement ----------------
X, Y = all_nodes[:,0], all_nodes[:,1]
ux_ana = -P * Y * (3*L - X) * X / (6*E*I)
uy_ana =  P * (3*L*X**2 - X**3) / (6*E*I)

# ---------------- Stress (numerical) ----------------
# helper to compute stress at element center

def element_center_stress(nodes, elems, Usub):
    sx, sy, txy, cx, cy = [], [], [], [], []
    for el in elems:
        coords = nodes[el]
        u_el = Usub[el].flatten()
        xi=eta=0
        dN_ref = dN_dxi_Q4(xi, eta)
        J = dN_ref @ coords
        invJ = np.linalg.inv(J)
        dN = invJ @ dN_ref
        B = np.zeros((3,8))
        for i in range(4):
            B[0,2*i]   = dN[0,i]
            B[1,2*i+1] = dN[1,i]
            B[2,2*i]   = dN[1,i]
            B[2,2*i+1] = dN[0,i]
        stress = D @ (B @ u_el)
        sx.append(stress[0]); sy.append(stress[1]); txy.append(stress[2])
        c = coords.mean(axis=0)
        cx.append(c[0]); cy.append(c[1])
    return np.array(sx), np.array(sy), np.array(txy), np.array(cx), np.array(cy)

sx1, sy1, t1, cx1, cy1 = element_center_stress(nodes1, elems1, U1)
sx2, sy2, t2, cx2, cy2 = element_center_stress(nodes2, elems2, U2)
num_sxx = np.concatenate([sx1, sx2])
cent_x  = np.concatenate([cx1, cx2])
cent_y  = np.concatenate([cy1, cy2])

# ---------------- Analytical σ_xx (beam theory) ----------------
ana_sxx = -P * cent_y * (L - cent_x) / I

# ---------------- Plot displacement & stress ----------------
fig, axs = plt.subplots(3,2, figsize=(14,14))
# displacements
axs[0,0].tricontourf(X, Y, ux_all, 20); axs[0,0].set_title("Numerical $u_x$")
axs[0,1].tricontourf(X, Y, ux_ana, 20); axs[0,1].set_title("Analytical $u_x$")
axs[1,0].tricontourf(X, Y, uy_all, 20); axs[1,0].set_title("Numerical $u_y$")
axs[1,1].tricontourf(X, Y, uy_ana, 20); axs[1,1].set_title("Analytical $u_y$")
# stress σ_xx
sc = axs[2,0].tricontourf(cent_x, cent_y, num_sxx, 20); axs[2,0].set_title("Numerical $σ_{xx}$")
plt.colorbar(sc, ax=axs[2,0])
sc2= axs[2,1].tricontourf(cent_x, cent_y, ana_sxx, 20); axs[2,1].set_title("Analytical $σ_{xx}$")
plt.colorbar(sc2, ax=axs[2,1])
for ax in axs.flat:
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect('equal'); ax.grid(True)
plt.tight_layout(); plt.show()
