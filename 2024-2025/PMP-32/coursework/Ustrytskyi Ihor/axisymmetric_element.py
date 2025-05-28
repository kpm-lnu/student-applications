from material_properties import Material
from shape_functions import ShapeFunction
from axisymmetric_quadrature import AxisymmetricQuadrature
import numpy as np

class AxisymmetricElement:
    def __init__(self, elem_id: int, node_ids: list[int], material: Material, 
                 shape_func: ShapeFunction, quadrature: AxisymmetricQuadrature):
        self.elem_id = elem_id
        self.node_ids = node_ids
        self.material = material
        self.shape_func = shape_func
        self.quadrature = quadrature

    def compute_element_stiffness(self, mesh: "Mesh"):
        D = self.material.get_elastic_matrix()

        dof = mesh.node_dof * self.shape_func.nodes_count
        Ke = np.zeros((dof, dof), dtype=float)

        element_nodes = [mesh.nodes[nid] for nid in self.node_ids]

        coords = np.array([[node.r, node.z] for node in element_nodes])

        for gp in self.quadrature.gauss_points():
            xi = gp["xi"]
            eta = gp["eta"]
            weight = gp["weight"]

            N, dN_dxi, dN_deta = self.shape_func.evaluate(xi, eta)

            J = np.zeros((mesh.node_dof, mesh.node_dof))
            for i in range(len(element_nodes)):
                J[0, 0] += dN_dxi[i] * coords[i, 0]
                J[0, 1] += dN_dxi[i] * coords[i, 1]
                J[1, 0] += dN_deta[i] * coords[i, 0]
                J[1, 1] += dN_deta[i] * coords[i, 1]

            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            dN_dr = np.zeros(len(element_nodes))
            dN_dz = np.zeros(len(element_nodes))
            for i in range(len(element_nodes)):
                dN_dr[i] = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i]
                dN_dz[i] = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i]
            
            r_current = sum(N[i] * coords[i, 0] for i in range(len(element_nodes)))

            strain_length = 4
            B = np.zeros((strain_length, dof))
            
            for i in range(len(element_nodes)):
                B[0, i] = dN_dr[i]
                B[1, len(element_nodes) + i] = dN_dz[i]
                B[2, i] = dN_dz[i]
                B[2, len(element_nodes) + i] = dN_dr[i]
                B[3, i] = N[i] / r_current

            dV = r_current * detJ * weight

            Ke += B.T @ D @ B * dV
        return Ke

