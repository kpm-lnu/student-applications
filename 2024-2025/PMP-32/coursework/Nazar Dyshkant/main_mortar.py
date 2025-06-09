import numpy as np
import matplotlib.pyplot as plt
from mesh_generator import generate_nonmatching_subdomains
from assembler_solver import assemble_global_stiffness, apply_dirichlet_bc, solve_system
from mortar import find_interface_nodes, extract_interface_edges

# Material properties
E = 210e9
nu = 0.3
D = (E / (1 - nu**2)) * np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1 - nu) / 2]
])

# Meshes for two subdomains
nodes1, elems1, nodes2, elems2, nodes_all, elems_all = generate_nonmatching_subdomains()

# Plot initial mesh (undeformed)
plt.figure(figsize=(8, 4))
for elem in elems1:
    coords = nodes1[elem]
    coords = np.vstack([coords, coords[0]])
    plt.plot(coords[:, 0], coords[:, 1], 'r--')
for elem in elems2:
    coords = nodes2[elem]
    coords = np.vstack([coords, coords[0]])
    plt.plot(coords[:, 0], coords[:, 1], 'b--')
plt.gca().set_aspect('equal')
plt.title("Initial Mesh (Left = Red, Right = Blue)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Show master interface nodes and their y-coordinates
iface_master = sorted(find_interface_nodes(nodes2), key=lambda i: nodes2[i][1])
print("\nMaster interface nodes and y-values:")
for i in iface_master:
    print(f"  Node {i:2d}  y = {nodes2[i][1]:.4f}")

# Assemble local stiffness matrices
K1 = assemble_global_stiffness(nodes1, elems1, D)
K2 = assemble_global_stiffness(nodes2, elems2, D)
F1 = np.zeros(K1.shape[0])
F2 = np.zeros(K2.shape[0])

# Apply uniform vertical load on right edge of right domain
tol = 1e-6
for i, (x, y) in enumerate(nodes2):
    if abs(x - 1.0) < tol:
        F2[2 * i + 1] = -1e5

# Apply full Dirichlet BC on left edge of left domain
fixed_dofs_1 = []
for i, (x, y) in enumerate(nodes1):
    if abs(x) < tol:
        fixed_dofs_1.extend([2 * i, 2 * i + 1])
K1, F1 = apply_dirichlet_bc(K1, F1, fixed_dofs_1)

# Find interface nodes (now treating nodes1 as slave, nodes2 as master)
iface_slave = sorted(find_interface_nodes(nodes1), key=lambda i: nodes1[i][1])

# Interpolate slave node displacements onto master side (x and y continuity)
L = len(iface_slave)
B1 = np.zeros((2 * L, 2 * len(nodes1)))
B2 = np.zeros((2 * L, 2 * len(nodes2)))

for i, s_idx in enumerate(iface_slave):
    y_slave = nodes1[s_idx][1]
    projected = False

    for j in range(len(iface_master) - 1):
        y0 = nodes2[iface_master[j]][1]
        y1 = nodes2[iface_master[j + 1]][1]
        if y0 <= y_slave <= y1:
            t = (y_slave - y0) / (y1 - y0)
            m0 = iface_master[j]
            m1 = iface_master[j + 1]

            # y continuity
            B2[2 * i + 1, 2 * m0 + 1] = 1 - t
            B2[2 * i + 1, 2 * m1 + 1] = t
            # x continuity
            B2[2 * i, 2 * m0] = 1 - t
            B2[2 * i, 2 * m1] = t

            projected = True
            print(f"Slave node {s_idx} at y = {y_slave:.4f} mapped between master nodes {m0}, {m1} with t = {t:.4f}")
            break

    if not projected:
        print(f"WARNING: Slave node {s_idx} at y = {y_slave:.4f} was not projected to master side!")

    # Enforce both x and y continuity
    B1[2 * i,     2 * s_idx]     = 1  # x-dof
    B1[2 * i + 1, 2 * s_idx + 1] = 1  # y-dof

# Build saddle-point system
n1 = K1.shape[0]
n2 = K2.shape[0]
K = np.block([
    [K1,                np.zeros((n1, n2)),   B1.T],
    [np.zeros((n2, n1)), K2,                 -B2.T],
    [B1,                -B2,                 np.zeros((2 * L, 2 * L))]
])

F = np.concatenate([F1, F2, np.zeros(2 * L)])
U = solve_system(K, F)

# Extract displacements
u1 = U[:n1]
u2 = U[n1:n1 + n2]

# Plot deformed mesh
scale = 1e3
plt.figure(figsize=(8, 4))
for elem in elems1:
    coords = nodes1[elem] + scale * np.column_stack((u1[2 * elem], u1[2 * elem + 1]))
    coords = np.vstack([coords, coords[0]])
    plt.plot(coords[:, 0], coords[:, 1], 'r')
for elem in elems2:
    coords = nodes2[elem] + scale * np.column_stack((u2[2 * elem], u2[2 * elem + 1]))
    coords = np.vstack([coords, coords[0]])
    plt.plot(coords[:, 0], coords[:, 1], 'b')
plt.gca().set_aspect('equal')
plt.title("Deformed Mesh with Mortar (u & v continuity, Scaled x1000)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

