import numpy as np


class TriangularMesh:
    def __init__(self, Lx, Ly, nx, ny):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny

        self.nodes, self.elements = self._generate()
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]

        self.boundary_nodes = self._find_boundary_nodes()
        self.boundary_edges = self._find_boundary_edges()

    def _generate(self):
        nx, ny = self.nx, self.ny
        Lx, Ly = self.Lx, self.Ly

        x = np.linspace(0, Lx, nx + 1)
        y = np.linspace(0, Ly, ny + 1)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        nodes = np.column_stack([xx.ravel(), yy.ravel()])

        elements = []
        for i in range(nx):
            for j in range(ny):
                n0 = i * (ny + 1) + j
                n1 = (i + 1) * (ny + 1) + j
                n2 = (i + 1) * (ny + 1) + (j + 1)
                n3 = i * (ny + 1) + (j + 1)
                elements.append([n0, n1, n3])
                elements.append([n1, n2, n3])

        return nodes, np.array(elements, dtype=int)

    def _find_boundary_nodes(self):
        tol = 1e-12
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        mask = (
            (np.abs(x) < tol) |
            (np.abs(x - self.Lx) < tol) |
            (np.abs(y) < tol) |
            (np.abs(y - self.Ly) < tol)
        )
        return np.where(mask)[0]

    def _find_boundary_edges(self):
        tol = 1e-12
        x, y = self.nodes[:, 0], self.nodes[:, 1]

        edge_count = {}
        for elem in self.elements:
            for k in range(3):
                e = tuple(sorted((elem[k], elem[(k + 1) % 3])))
                edge_count[e] = edge_count.get(e, 0) + 1

        boundary = [e for e, c in edge_count.items() if c == 1]

        sides = {'left': [], 'right': [], 'bottom': [], 'top': []}
        for n1, n2 in boundary:
            xm = 0.5 * (x[n1] + x[n2])
            ym = 0.5 * (y[n1] + y[n2])
            if abs(xm) < tol:
                sides['left'].append((n1, n2))
            elif abs(xm - self.Lx) < tol:
                sides['right'].append((n1, n2))
            elif abs(ym) < tol:
                sides['bottom'].append((n1, n2))
            elif abs(ym - self.Ly) < tol:
                sides['top'].append((n1, n2))

        return {k: np.array(v, dtype=int) for k, v in sides.items()}

    def element_nodes(self, e):
        return self.nodes[self.elements[e]]

    def element_area(self, e):
        coords = self.element_nodes(e)
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    def hmax(self):
        h = 0.0
        for e in range(self.n_elements):
            coords = self.element_nodes(e)
            for i in range(3):
                for j in range(i + 1, 3):
                    d = np.linalg.norm(coords[i] - coords[j])
                    h = max(h, d)
        return h

    def hmin(self):
        h = np.inf
        for e in range(self.n_elements):
            coords = self.element_nodes(e)
            for i in range(3):
                for j in range(i + 1, 3):
                    d = np.linalg.norm(coords[i] - coords[j])
                    h = min(h, d)
        return h
