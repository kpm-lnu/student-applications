import numpy as np
import matplotlib.pyplot as plt
from mesh_generator import generate_single_domain
from assembler_solver import assemble_global_stiffness, apply_dirichlet_bc, solve_system

# Material properties (plane stress)
E = 210e9      # Young's modulus in Pascals
nu = 0.3       # Poisson's ratio
D = (E / (1 - nu**2)) * np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1 - nu) / 2]
])

# Mesh
nodes, elements = generate_single_domain()
num_nodes = nodes.shape[0]

# Assemble global stiffness matrix
K = assemble_global_stiffness(nodes, elements, D)
F = np.zeros(2 * num_nodes)  # Global force vector

# Apply external force (e.g., downward force on right edge)
tol = 1e-6
for i, (x, y) in enumerate(nodes):
    if abs(x - 1.0) < tol:
        F[2 * i + 1] = -1e5  # Apply -100 kN in y-direction

# Apply Dirichlet boundary conditions (fix left edge)
fixed_dofs = []
for i, (x, y) in enumerate(nodes):
    if abs(x) < tol:
        fixed_dofs.extend([2 * i, 2 * i + 1])

K_bc, F_bc = apply_dirichlet_bc(K.copy(), F.copy(), fixed_dofs)

# Solve the system
U = solve_system(K_bc, F_bc)

# Plot deformed mesh (scaled)
scale = 1e3
plt.figure(figsize=(8, 8))
for elem in elements:
    pts = nodes[elem]
    u = U[2 * np.array(elem)]
    v = U[2 * np.array(elem) + 1]
    displaced_pts = pts + scale * np.column_stack((u, v))

    # original in gray
    poly = np.append(elem, elem[0])
    plt.plot(nodes[poly, 0], nodes[poly, 1], 'gray', linestyle='--')

    # deformed in blue
    plt.plot(displaced_pts[:, 0].tolist() + [displaced_pts[0, 0]],
             displaced_pts[:, 1].tolist() + [displaced_pts[0, 1]], 'b')

plt.gca().set_aspect('equal')
plt.title("Deformed Mesh (scaled by 1000x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()