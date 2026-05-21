from dataclasses import dataclass
import numpy as np


@dataclass
class TriangleElementData:
    area: float
    B: np.ndarray
    dofs: np.ndarray


def triangle_B_matrix(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Обчислює площу та матрицю B для лінійного трикутного елемента.

    coords: масив 3x2 з координатами вузлів елемента.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * abs(det)

    if area <= 1e-14:
        raise ValueError("Вироджений трикутний елемент із нульовою площею.")

    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float)

    B = np.zeros((3, 6), dtype=float)
    for i in range(3):
        B[0, 2 * i] = b[i]
        B[1, 2 * i + 1] = c[i]
        B[2, 2 * i] = c[i]
        B[2, 2 * i + 1] = b[i]

    B /= (2.0 * area)
    return area, B


def element_dofs(element_nodes: np.ndarray) -> np.ndarray:
    """Повертає глобальні номери ступенів вільності для трикутного елемента."""
    dofs = []
    for node in element_nodes:
        dofs.extend([2 * int(node), 2 * int(node) + 1])
    return np.array(dofs, dtype=int)


def local_stiffness(coords: np.ndarray, D: np.ndarray, thickness: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Обчислює локальну матрицю жорсткості K_e."""
    area, B = triangle_B_matrix(coords)
    Ke = thickness * area * (B.T @ D @ B)
    return Ke, area, B
