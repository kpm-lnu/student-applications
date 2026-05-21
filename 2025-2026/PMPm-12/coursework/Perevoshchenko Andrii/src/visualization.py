import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def plot_mesh(mesh, title="Triangular Mesh", figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    ax.triplot(triang, 'k-', linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig, ax


def plot_solution(mesh, u, title="FEM Solution", figsize=(10, 7),
                  levels=20, cmap='viridis', show_mesh=False):
    fig, ax = plt.subplots(figsize=figsize)
    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)

    tcf = ax.tricontourf(triang, u, levels=levels, cmap=cmap)
    if show_mesh:
        ax.triplot(triang, 'k-', linewidth=0.3, alpha=0.3)
    fig.colorbar(tcf, ax=ax, label='u')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig, ax


def plot_solution_3d(mesh, u, title="FEM Solution (3D)", figsize=(10, 7),
                     cmap='viridis'):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    ax.plot_trisurf(triang, u, cmap=cmap, edgecolor='none', alpha=0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_snapshots(mesh, solutions, times, snapshot_times,
                   figsize=(16, 4), levels=20, cmap='viridis'):
    times = np.array(times)
    n_snaps = len(snapshot_times)
    fig, axes = plt.subplots(1, n_snaps, figsize=figsize)
    if n_snaps == 1:
        axes = [axes]

    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)

    vmin = min(np.min(s) for s in solutions)
    vmax = max(np.max(s) for s in solutions)
    lvls = np.linspace(vmin, vmax, levels)

    for ax, snap_t in zip(axes, snapshot_times):
        idx = np.argmin(np.abs(times - snap_t))
        tcf = ax.tricontourf(triang, solutions[idx], levels=lvls, cmap=cmap)
        ax.set_title(f't = {times[idx]:.4f}')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.colorbar(tcf, ax=axes, label='u', shrink=0.8)
    plt.tight_layout()
    return fig


def animate_solution(mesh, solutions, times, figsize=(10, 7),
                     levels=20, cmap='viridis', interval=100):
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=figsize)
    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)

    vmin = min(np.min(s) for s in solutions)
    vmax = max(np.max(s) for s in solutions)
    lvls = np.linspace(vmin, vmax, levels)

    tcf = ax.tricontourf(triang, solutions[0], levels=lvls, cmap=cmap)
    fig.colorbar(tcf, ax=ax, label='u')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_title(f't = {times[0]:.4f}')

    def update(frame):
        ax.clear()
        ax.tricontourf(triang, solutions[frame], levels=lvls, cmap=cmap)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_title(f't = {times[frame]:.4f}')
        return []

    ani = FuncAnimation(fig, update, frames=len(solutions),
                        interval=interval, blit=False)
    plt.tight_layout()
    return ani
