import triangle as tr
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

from file1 import (
    L2_Error,
    assemble_rhs,
    assemble_global_matrix,
    compute_qe,
    compute_ke,
    exact,
)
from plotter import plot_mesh

rectangle_vertices = [[0, 0], [3, 0], [0, 1], [3, 1]]

A = dict(
    vertices=np.array(rectangle_vertices), segments=tr.convex_hull(rectangle_vertices)
)
B = tr.triangulate(A, "Da0.08 q30p")

plot_mesh(B)

vertices = B["vertices"]
triangles = B["triangles"]
triangle_vertices = np.array([[vertices[j] for j in i] for i in triangles])

assembled_system = np.zeros((len(vertices), len(vertices)))
rhs = np.zeros(len(vertices))

for i in range(len(triangles)):
    ke = np.array(compute_ke(triangle_vertices[i], a11=1, a22=1))
    qe = compute_qe(triangle_vertices[i], fe=[3, 3, 3])
    assembled_system = assemble_global_matrix(assembled_system, ke, triangles[i])
    rhs = assemble_rhs(qe, triangles[i], rhs)

for i in range(len(vertices)):
    if vertices[i, 0] == 0 or vertices[i, 0] == 3:
        assembled_system[i, i] = 1000000000000

solution = np.linalg.solve(assembled_system, rhs)

u_exact = [exact(vertices[i, 0]) for i in range(len(vertices))]
error_L2 = L2_Error(solution, u_exact)
abs_errors = [[i + 1, abs(u_exact[i] - solution[i])] for i in range(len(solution))]

print(tabulate(abs_errors, headers=["#", "Absolute error"]))
print("L2 Error", error_L2)

# TASK2

A = dict(
    vertices=np.array(rectangle_vertices), segments=tr.convex_hull(rectangle_vertices)
)
B = tr.triangulate(A, "Da0.016 q30p")

plot_mesh(B)

vertices = B["vertices"]
triangles = B["triangles"]
triangle_vertices = np.array([[vertices[j] for j in i] for i in triangles])

assembled_system = np.zeros((len(vertices), len(vertices)))
rhs = np.zeros(len(vertices))

for i in range(len(triangles)):
    ke = np.array(compute_ke(triangle_vertices[i], a11=0, a22=-1))
    qe = compute_qe(triangle_vertices[i], fe=[3, 3, 3])
    assembled_system = assemble_global_matrix(assembled_system, ke, triangles[i])
    rhs = assemble_rhs(qe, triangles[i], rhs)

for i in range(len(vertices)):
    if vertices[i, 0] == 0 or vertices[i, 0] == 3:
        assembled_system[i, i] = 1000000000000

solution = np.linalg.solve(assembled_system, rhs)

u_exact2 = [exact(vertices[i, 0]) for i in range(len(vertices))]
error_L22 = L2_Error(solution, u_exact2)
abs_errors2 = [[i + 1, abs(u_exact2[i] - solution[i])] for i in range(len(solution))]

print(tabulate(abs_errors2, headers=["3", "Absolute error"]))
print("L2 Error", error_L22)

X = B["vertices"][:, 0]
Y = B["vertices"][:, 1]
Z = solution
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_trisurf(X, Y, Z, cmap="viridis", edgecolor="none")
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title("3D u(x, y)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("u(x, y)")
plt.show()
