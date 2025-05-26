from node import Node
from element import AxisymmetricElement

class Mesh:
    def __init__(self, material, shape_func, quadrature, node_dof):
        self.nodes = {}
        self.elements = {}
        self.material = material
        self.shape_func = shape_func
        self.quadrature = quadrature
        self.node_dof = node_dof
    
    def add_node(self, node: Node):
        self.nodes[node.node_id] = node

    def add_element(self, element: AxisymmetricElement):
        self.elements[element.elem_id] = element

    def generate_rectangles(self, r_min, r_max, z_min, z_max, rN, zN):
        dr = (r_max - r_min) / rN
        dz = (z_max - z_min) / zN

        node_id = 0
        for j in range(zN + 1):
            for i in range(rN + 1):
                r = r_min + i * dr
                z = z_min + j * dz
                self.add_node(Node(node_id, r, z))
                node_id += 1

        elem_id = 0
        for j in range(zN):
            for i in range(rN):
                n1 = j * (rN + 1) + i
                n2 = n1 + 1
                n3 = n1 + (rN + 1)
                n4 = n3 + 1
                self.add_element(AxisymmetricElement(elem_id, [n1, n2, n4, n3], self.material, self.shape_func, self.quadrature))
                elem_id += 1