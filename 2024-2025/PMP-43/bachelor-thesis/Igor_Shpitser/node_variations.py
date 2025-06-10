from shapeFunction import LinearQuadrilateralShapeFunction
from axisymmetricQuadrature import AxisymmetricQuadrature
from element import AxisymmetricElement
from mesh import Mesh
from node import Node
import numpy as np

def is_convex(quad):
    """Перевіряє, чи є чотирикутник опуклим."""
    cross_products = []
    for i in range(4):
        p1, p2, p3 = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        v1, v2 = p2 - p1, p3 - p2
        cross_product = np.cross(v1, v2)
        cross_products.append(cross_product)

    return all(c > 0 for c in cross_products) or all(c < 0 for c in cross_products)


def are_collinear(quad):
    """Перевіряє, чи лежать всі точки на одній прямій."""
    p1, p2 = quad[0], quad[1]
    base_vector = p2 - p1

    for i in range(2, 4):
        vector = quad[i] - p1
        if np.cross(base_vector, vector) != 0:
            return False

    return True


def validate_quadrilateral(quad):
    """Перевіряє опуклість та колінеарність."""
    if not is_convex(quad):
        raise ValueError("Чотирикутник не є опуклим!")
    if are_collinear(quad):
        raise ValueError("Усі точки чотирикутника лежать на одній прямій!")


def process_quadrilateral(quadrilateral, num_variations=100000, noise_level=3.0):
    """
    Нормалізує чотирикутник, коригує скошення та генерує варіації.
    """
    quad = np.array(quadrilateral, dtype=np.float64)

    # Перевірка
    validate_quadrilateral(quad)

    # # Нормалізація
    # center = np.mean(quad, axis=0)
    # quad -= center
    #
    # distances = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    # longest_idx = np.argmax(distances)
    # p1, p2 = quad[longest_idx], quad[(longest_idx + 1) % 4]
    # angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    # rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],
    #                             [np.sin(-angle), np.cos(-angle)]])
    # quad = np.dot(quad, rotation_matrix.T)
    #
    # max_length = np.max(distances)
    # quad /= max_length

    # # Коригування скошення (пропорційне збільшення меншої сторони)
    # min_side_idx = np.argmin(distances)
    # scale_factor = max_length / distances[min_side_idx]
    # quad[min_side_idx] *= scale_factor

    # Генерація варіацій
    # fixed_indices = np.random.choice(4, 2, replace=False)  # Фіксуємо 2 точки
    free_indices = [1,2]

    variations = []
    for _ in range(num_variations):
        new_quad = quad.copy()

        # Додаємо шум тільки для вільних точок
        for idx in free_indices:
            noise = np.random.uniform(0, noise_level, size=2)
            new_quad[idx] *= noise

        if is_convex(new_quad) and not are_collinear(new_quad):
            variations.append(new_quad.tolist())

    print(f"Generated {len(variations)} valid variations")
    return quad.tolist(), variations

def compute_fitness_for_variation(variation_coords, shape_func=None, quadrature=None, Ke_ref=None):
    # Параметри
    shape_func = shape_func or LinearQuadrilateralShapeFunction()
    quadrature = quadrature or AxisymmetricQuadrature(n_points=2)
    node_dof = 2

    # Створення вузлів
    mesh = Mesh(material=None, shape_func=shape_func, quadrature=quadrature, node_dof=node_dof)
    for i, (r, z) in enumerate(variation_coords):
        node = Node(node_id=i, r=r, z=z)
        mesh.add_node(node)

    # Створення елемента (використовуємо всі 4 вузли)
    elem = AxisymmetricElement(elem_id=0, node_ids=[0, 1, 2, 3], material=None,
                               shape_func=shape_func, quadrature=quadrature)

    Ke = elem.compute_element_stiffness(mesh)

    if Ke_ref is not None:
        # Фітнес: нормована різниця
        fitness = np.linalg.norm(Ke - Ke_ref) / np.linalg.norm(Ke_ref)
        return fitness, Ke
    else:
        return Ke

if __name__ == "__main__":
    quad = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    normalized_quad, variations = process_quadrilateral(quad, num_variations=10)

    for i, variation in enumerate(variations[:3]):
        fitness, Ke = compute_fitness_for_variation(variation, Ke_ref=np.eye(8))  # умовний Ke_ref
        print(f"Variation {i} - fitness: {fitness:.5e}")

# import numpy as np
#
# def generate_variations(quadrilateral, num_variations=100000, noise_level=0.25):
#     variations = []
#     quad = np.array(quadrilateral, dtype=np.float64)
#     center = np.mean(quad, axis=0)
#
#     for _ in range(num_variations):
#         noise = np.random.uniform(-noise_level, noise_level, quad.shape)
#         new_quad = quad + noise  # Додаємо випадковий шум
#         variations.append(new_quad.tolist())
#
#     print(f"Generated {len(variations)} variations")
#     return variations
#
#
# def normalize_quadrilateral(quadrilateral):
#     quad = np.array(quadrilateral, dtype=np.float64)
#     center = np.mean(quad, axis=0)
#     quad -= center
#
#     distances = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
#     longest_idx = np.argmax(distances)
#     p1, p2 = quad[longest_idx], quad[(longest_idx + 1) % 4]
#     angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
#     rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],
#                                 [np.sin(-angle), np.cos(-angle)]])
#     quad = np.dot(quad, rotation_matrix.T)
#
#     max_length = np.max(distances)
#     quad /= max_length
#
#     return quad.tolist()
