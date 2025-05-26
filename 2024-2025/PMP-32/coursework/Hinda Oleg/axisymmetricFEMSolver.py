from mesh import Mesh
from boundaryConditions import BoundaryCondition
import numpy as np
from timingUtils import stopwatch

class AxisymmetricFEMSolver:
    def __init__(self, mesh: Mesh, bcs: list[BoundaryCondition]):
        self.mesh = mesh
        self.bcs = bcs
        
        self.global_stiffness = None
        self.global_force = None
        self.displacements = None

    def assign_dof_indices(self):
        num_nodes = len(self.mesh.nodes)

        for node_id, node in self.mesh.nodes.items():
            node.dof_indices = [node_id, num_nodes + node_id]


    def assemble_system(self):
        n_dof = self.mesh.node_dof * len(self.mesh.nodes)
        self.global_stiffness = np.zeros((n_dof, n_dof), dtype=float)
        self.global_force = np.zeros(n_dof, dtype=float)
        
        for elem_id, elem in self.mesh.elements.items():
            Ke = elem.compute_element_stiffness(self.mesh)
            
            num_nodes = len(self.mesh.nodes)
            dof_map = [None] * (self.mesh.node_dof * len(elem.node_ids))
            for a, node_id in enumerate(elem.node_ids):
                dof_map[a] = node_id
                dof_map[len(elem.node_ids) + a] = node_id + num_nodes
        
            for i_local, i_global in enumerate(dof_map):
                for j_local, j_global in enumerate(dof_map):
                    self.global_stiffness[i_global, j_global] += Ke[i_local, j_local]

    def apply_boundary_conditions(self):
        for bc in self.bcs:
            bc.apply(self.global_stiffness, self.global_force, self.mesh)

    @stopwatch
    def solve(self):
        self.displacements = np.linalg.solve(self.global_stiffness, self.global_force)
        
        num_nodes = len(self.mesh.nodes)

        for node_id, node in self.mesh.nodes.items():
            node.displacements[0] = self.displacements[node_id]
            node.displacements[1] = self.displacements[num_nodes + node_id]

    def run(self):
        self.assign_dof_indices()
        self.assemble_system()
        self.apply_boundary_conditions()
        self.solve()
        print("FEM solution complete.")
