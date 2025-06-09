import numpy as np
import matplotlib.pyplot as plt

def generate_rect_mesh(x_start, x_end, y_start, y_end, nx, ny):
    """
    Generates a structured rectangular mesh with Q4 (4-node) elements.
    Returns:
        nodes: (N, 2) array of node coordinates
        elements: (M, 4) array of element connectivity (node indices)
    """
    x = np.linspace(x_start, x_end, nx + 1)
    y = np.linspace(y_start, y_end, ny + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack((xx.ravel(), yy.ravel()))

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            elements.append([n0, n1, n2, n3])

    return np.array(nodes), np.array(elements)

# def generate_nonmatching_subdomains():
#     """
#     Generate two subdomains:
#         Omega1: [0, 0.5] x [0, 1], mesh size 4x6
#         Omega2: [0.5, 1] x [0, 1], mesh size 6x8
#     Note: meshes do not match along the interface x = 0.5
#     """
#     nodes1, elems1 = generate_rect_mesh(0.0, 0.5, 0.0, 1.0, nx=4, ny=6)
#     nodes2, elems2 = generate_rect_mesh(0.5, 1.0, 0.0, 1.0, nx=6, ny=8)
#
#     # Adjust node indices in elems2
#     offset = nodes1.shape[0]
#     elems2_offset = elems2 + offset
#     nodes_combined = np.vstack((nodes1, nodes2))
#     elems_combined = np.vstack((elems1, elems2_offset))
#
#     return nodes1, elems1, nodes2, elems2, nodes_combined, elems_combined

def generate_nonmatching_subdomains(nx1=4, ny1=6, nx2=6, ny2=8):
    nodes1, elems1 = generate_rect_mesh(0.0, 0.5, 0.0, 1.0, nx=nx1, ny=ny1)
    nodes2, elems2 = generate_rect_mesh(0.5, 1.0, 0.0, 1.0, nx=nx2, ny=ny2)

    offset = nodes1.shape[0]
    elems2_offset = elems2 + offset
    nodes_combined = np.vstack((nodes1, nodes2))
    elems_combined = np.vstack((elems1, elems2_offset))

    return nodes1, elems1, nodes2, elems2, nodes_combined, elems_combined


if __name__ == "main":
    nodes1, elems1, nodes2, elems2, nodes_all, elems_all = generate_nonmatching_subdomains()

    def plot_mesh(nodes, elements, color='b'):
        for elem in elements:
            poly = np.append(elem, elem[0])
            plt.plot(nodes[poly, 0], nodes[poly, 1], color)

    plt.figure(figsize=(8, 4))
    plot_mesh(nodes1, elems1, color='r')
    plot_mesh(nodes2, elems2, color='b')
    plt.gca().set_aspect('equal')
    plt.title("Non-Matching Meshes: Left (Red), Right (Blue)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()