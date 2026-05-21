from dataclasses import dataclass
import numpy as np
from scipy.spatial import Delaunay


@dataclass
class Mesh:
    nodes: np.ndarray
    elements: np.ndarray
    boundary_edges: list[tuple[int, int]]


def _boundary_edges(elements: np.ndarray) -> list[tuple[int, int]]:
    edge_count: dict[tuple[int, int], int] = {}
    for tri in elements:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for a, b in edges:
            edge = tuple(sorted((int(a), int(b))))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    return [edge for edge, count in edge_count.items() if count == 1]


def rectangular_tri_mesh(L: float, H: float, nx: int, ny: int) -> Mesh:
    """Структурована трикутна сітка прямокутника [0,L]x[0,H]."""
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, H, ny + 1)

    nodes = []
    for y in ys:
        for x in xs:
            nodes.append([x, y])
    nodes = np.array(nodes, dtype=float)

    def idx(i: int, j: int) -> int:
        return j * (nx + 1) + i

    elements = []
    for j in range(ny):
        for i in range(nx):
            n00 = idx(i, j)
            n10 = idx(i + 1, j)
            n01 = idx(i, j + 1)
            n11 = idx(i + 1, j + 1)
            elements.append([n00, n10, n11])
            elements.append([n00, n11, n01])

    elements = np.array(elements, dtype=int)
    return Mesh(nodes=nodes, elements=elements, boundary_edges=_boundary_edges(elements))


def quarter_plate_with_hole_mesh(L: float, H: float, R: float, n_rect: int = 28, n_theta: int = 40, n_radial: int = 10) -> Mesh:
    """
    Неструктурована сітка для чверті пластини з круглим отвором.
    Область: 0<=x<=L/2, 0<=y<=H/2, x^2+y^2>=R^2.
    """
    xmax = L / 2.0
    ymax = H / 2.0
    pts = []

    xs = np.linspace(0.0, xmax, n_rect + 1)
    ys = np.linspace(0.0, ymax, n_rect + 1)
    for x in xs:
        for y in ys:
            if x * x + y * y >= (R * 0.985) ** 2:
                pts.append([x, y])

    thetas = np.linspace(0.0, np.pi / 2.0, n_theta + 1)
    for k in range(n_radial + 1):
        s = k / max(n_radial, 1)
        rmax = min(xmax, ymax) * 0.95
        r = R + (rmax - R) * (s ** 1.6)
        for th in thetas:
            x, y = r * np.cos(th), r * np.sin(th)
            if x <= xmax + 1e-12 and y <= ymax + 1e-12:
                pts.append([x, y])

    for x in xs:
        pts.append([x, ymax])
    for y in ys:
        pts.append([xmax, y])

    pts = np.unique(np.round(np.array(pts, dtype=float), decimals=12), axis=0)
    tri = Delaunay(pts)
    elements = []

    for simplex in tri.simplices:
        c = pts[simplex].mean(axis=0)
        r = np.linalg.norm(c)
        if (
            c[0] >= -1e-12 and c[0] <= xmax + 1e-12 and
            c[1] >= -1e-12 and c[1] <= ymax + 1e-12 and
            r >= R - 1e-12
        ):
            edge_lengths = [
                np.linalg.norm(pts[simplex[0]] - pts[simplex[1]]),
                np.linalg.norm(pts[simplex[1]] - pts[simplex[2]]),
                np.linalg.norm(pts[simplex[2]] - pts[simplex[0]]),
            ]
            if max(edge_lengths) <= max(xmax, ymax) / 3.0:
                elements.append(simplex)

    elements = np.array(elements, dtype=int)
    return Mesh(nodes=pts, elements=elements, boundary_edges=_boundary_edges(elements))


def quarter_ring_mesh(ri: float, ro: float, nr: int = 14, nt: int = 40) -> Mesh:
    """Структурована трикутна сітка чверті товстостінного кільця."""
    rs = np.linspace(ri, ro, nr + 1)
    thetas = np.linspace(0.0, np.pi / 2.0, nt + 1)

    nodes = []
    for r in rs:
        for th in thetas:
            nodes.append([r * np.cos(th), r * np.sin(th)])
    nodes = np.array(nodes, dtype=float)

    def idx(i: int, j: int) -> int:
        return i * (nt + 1) + j

    elements = []
    for i in range(nr):
        for j in range(nt):
            n00 = idx(i, j)
            n01 = idx(i, j + 1)
            n10 = idx(i + 1, j)
            n11 = idx(i + 1, j + 1)
            elements.append([n00, n10, n11])
            elements.append([n00, n11, n01])

    elements = np.array(elements, dtype=int)
    return Mesh(nodes=nodes, elements=elements, boundary_edges=_boundary_edges(elements))


def nodes_on_x(mesh: Mesh, x_value: float, tol: float = 1e-9) -> np.ndarray:
    return np.where(np.isclose(mesh.nodes[:, 0], x_value, atol=tol))[0]


def nodes_on_y(mesh: Mesh, y_value: float, tol: float = 1e-9) -> np.ndarray:
    return np.where(np.isclose(mesh.nodes[:, 1], y_value, atol=tol))[0]


def nodes_on_radius(mesh: Mesh, radius: float, tol: float = 1e-8) -> np.ndarray:
    r = np.linalg.norm(mesh.nodes, axis=1)
    return np.where(np.isclose(r, radius, atol=tol))[0]


def classify_edges_by_midpoint(mesh: Mesh, predicate) -> list[tuple[int, int]]:
    result = []
    for a, b in mesh.boundary_edges:
        mid = 0.5 * (mesh.nodes[a] + mesh.nodes[b])
        if predicate(mid):
            result.append((a, b))
    return result
