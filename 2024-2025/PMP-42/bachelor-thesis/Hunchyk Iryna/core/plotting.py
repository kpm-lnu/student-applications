import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_mesh(nodes, elements, save_path=None, show=True):
    """
    Візуалізація трикутної сітки.

    :param nodes: список координат [(x₀, y₀), ...]
    :param elements: список трикутників [(i₁, i₂, i₃), ...]
    :param save_path: шлях до збереження графіку (якщо потрібно)
    :param show: чи показувати графік у вікні
    """
    fig, ax = plt.subplots()
    for tri in elements:
        pts = np.array([nodes[i] for i in tri + (tri[0],)])
        ax.plot(pts[:, 0], pts[:, 1], 'k-')

    ax.set_aspect('equal')
    ax.set_title("Сітка (triangular mesh)")

    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_trisurf(nodes, elements, values, save_path=None, show=True):
    """
    Побудова 3D-графіку функції u(x, y) по вузлах.

    :param nodes: координати вузлів
    :param elements: список трикутників
    :param values: значення u в кожному вузлі (u[i])
    :param save_path: шлях для збереження зображення
    :param show: чи показувати
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pts = np.array(nodes)
    x = pts[:, 0]
    y = pts[:, 1]
    z = np.array(values)

    triangles = np.array(elements)

    ax.plot_trisurf(x, y, z, triangles=triangles, cmap='viridis', edgecolor='none')
    ax.set_title("Чисельне розв’язання u(x, y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")

    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
