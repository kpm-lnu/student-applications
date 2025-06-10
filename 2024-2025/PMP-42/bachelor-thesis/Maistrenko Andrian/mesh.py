from node import Node
from element import AxisymmetricElement
import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    def __init__(self, material, shape_func, quadrature, node_dof):
        self.nodes = {}
        self.elements = {}
        self.material = material
        self.shape_func = shape_func
        self.quadrature = quadrature
        self.node_dof = node_dof
        self.boundary_nodes = []
    
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
    
    def generate_skewed_rectangle(self, corner_points, rN, zN):

        if len(corner_points) != 4:
            raise ValueError("Скошений прямокутник повинен мати рівно 4 кутові точки")
        
        # Отримання кутових точок
        p0 = np.array(corner_points[0])  # Нижній лівий кут
        p1 = np.array(corner_points[1])  # Нижній правий кут
        p2 = np.array(corner_points[2])  # Верхній правий кут
        p3 = np.array(corner_points[3])  # Верхній лівий кут
        
        node_id = 0
        node_ids = np.zeros((zN + 1, rN + 1), dtype=int)
        
        for j in range(zN + 1):
            for i in range(rN + 1):
                xi = i / rN
                eta = j / zN
                
                point = (1 - xi) * (1 - eta) * p0 + \
                        xi * (1 - eta) * p1 + \
                        xi * eta * p2 + \
                        (1 - xi) * eta * p3
                
                r, z = point
                self.add_node(Node(node_id, r, z))
                
                node_ids[j, i] = node_id
                
                if i == 0 or i == rN or j == 0 or j == zN:
                    self.boundary_nodes.append(node_id)
                
                node_id += 1
        
        elem_id = 0
        for j in range(zN):
            for i in range(rN):
                n1 = node_ids[j, i]
                n2 = node_ids[j, i+1]
                n3 = node_ids[j+1, i+1]
                n4 = node_ids[j+1, i]
                self.add_element(AxisymmetricElement(elem_id, [n1, n2, n3, n4], self.material, self.shape_func, self.quadrature))
                elem_id += 1
    
    def generate_skewed_cylinder(self, r_min, r_max, z_min, z_max, skew_angle_degrees, rN, zN):
        skew_angle = np.radians(skew_angle_degrees)
        
        max_skew = (r_max - r_min) * np.tan(skew_angle)
        
        node_id = 0
        node_ids = np.zeros((zN + 1, rN + 1), dtype=int)
        
        for j in range(zN + 1):
            z_ratio = (j / zN) if zN > 0 else 0
            current_skew = max_skew * z_ratio
            
            for i in range(rN + 1):
                r_ratio = i / rN
                r_base = r_min + r_ratio * (r_max - r_min)
                
                r_skew = current_skew * r_ratio
                
                r = r_base + r_skew
                z = z_min + z_ratio * (z_max - z_min)
                
                self.add_node(Node(node_id, r, z))
                node_ids[j, i] = node_id
                
                if i == 0 or i == rN or j == 0 or j == zN:
                    self.boundary_nodes.append(node_id)
                
                node_id += 1
        
        elem_id = 0
        for j in range(zN):
            for i in range(rN):
                n1 = node_ids[j, i]
                n2 = node_ids[j, i+1]
                n3 = node_ids[j+1, i+1]
                n4 = node_ids[j+1, i]
                self.add_element(AxisymmetricElement(elem_id, [n1, n2, n3, n4], self.material, self.shape_func, self.quadrature))
                elem_id += 1
                
    def visualize_mesh(self):
        plt.figure(figsize=(10, 8))
        
        for node_id, node in self.nodes.items():
            if node_id in self.boundary_nodes:
                plt.plot(node.r, node.z, 'ro', markersize=4)  # Граничні вузли червоним
            else:
                plt.plot(node.r, node.z, 'bo', markersize=2)  # Внутрішні вузли синім
        
        for elem_id, elem in self.elements.items():
            node_ids = elem.node_ids  # Змінено з elem.nodes на elem.node_ids
            for i in range(len(node_ids)):
                n1 = self.nodes[node_ids[i]]
                n2 = self.nodes[node_ids[(i+1) % len(node_ids)]]
                plt.plot([n1.r, n2.r], [n1.z, n2.z], 'k-', linewidth=0.5)
        
        plt.axis('equal')
        plt.grid(True)
        plt.title('Сітка прямокутних елементів')
        plt.xlabel('r')
        plt.ylabel('z')
        plt.show()