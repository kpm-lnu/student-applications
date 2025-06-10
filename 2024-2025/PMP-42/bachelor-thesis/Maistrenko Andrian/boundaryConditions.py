import numpy as np
from numpy.polynomial.legendre import leggauss

class BoundaryCondition:
    def apply(self, global_stiffness, global_force, mesh):
        raise NotImplementedError("Must implement BoundaryCondition apply method.")


class DirichletBC(BoundaryCondition):
    def __init__(self, node_id, dof, value):
        self.node_id = node_id
        self.dof = dof
        self.value = value

    def apply(self, global_stiffness, global_force, mesh):
        node = mesh.nodes[self.node_id]
        global_dof = node.dof_indices[self.dof]
        
        size = global_stiffness.shape[0]
        for i in range(size):
            if i == global_dof:
                global_stiffness[global_dof, global_dof] = 1.0
            else:
                global_stiffness[global_dof, i] = 0.0
                global_stiffness[i, global_dof] = 0.0
        
        global_force[global_dof] = self.value

class NeumannBC(BoundaryCondition):
    def __init__(self, edge_nodes: list[int], dof: int, pressure: float):
        self.edge_nodes = edge_nodes
        self.dof = dof
        self.pressure = pressure

    def apply(self, global_stiffness, global_force, mesh):
        xi_gp, w_gp = leggauss(2)

        for i in range(len(self.edge_nodes) - 1):
            node1 = mesh.nodes[self.edge_nodes[i]]
            node2 = mesh.nodes[self.edge_nodes[i+1]]
            r1, z1 = node1.r, node1.z
            r2, z2 = node2.r, node2.z

            L = np.sqrt((r2 - r1)**2 + (z2 - z1)**2)

            for xi, weight in zip(xi_gp, w_gp):
                N1 = 0.5 * (1 - xi)
                N2 = 0.5 * (1 + xi)

                r_gp = N1 * r1 + N2 * r2

                dS = (L / 2) * weight

                traction = self.pressure * r_gp

                global_dof1 = node1.dof_indices[self.dof]
                global_dof2 = node2.dof_indices[self.dof]
                global_force[global_dof1] += N1 * traction * dS
                global_force[global_dof2] += N2 * traction * dS