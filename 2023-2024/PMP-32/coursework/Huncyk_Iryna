import matplotlib.pyplot as plt


def plot_mesh(mesh):
    vertices = mesh['vertices']
    triangles = mesh['triangles']

    plt.figure()
    plt.gca().set_aspect('equal')
    plt.triplot(vertices[:, 0], vertices[:, 1], triangles, 'go-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Triangulated Mesh')
    plt.show()
