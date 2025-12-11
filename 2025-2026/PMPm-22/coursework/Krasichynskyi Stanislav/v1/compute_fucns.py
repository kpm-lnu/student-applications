import numpy as np


def compute_ke(triangle_vertices, a_11, a_22):
    i = 0
    j = 1
    m = 2
    x1 = 0
    x2 = 1
    ke = [
        [
            a_11 * (triangle_vertices[j, x2] - triangle_vertices[m, x2]) ** 2
            + a_22 * (triangle_vertices[m, x1] - triangle_vertices[j, x1]) ** 2,
            a_11
            * (triangle_vertices[j, x2] - triangle_vertices[m, x2])
            * (triangle_vertices[m, x2] - triangle_vertices[i, x2])
            + a_22
            * (triangle_vertices[m, x1] - triangle_vertices[j, x1])
            * (triangle_vertices[i, x1] - triangle_vertices[m, x1]),
            a_11
            * (triangle_vertices[j, x2] - triangle_vertices[m, x2])
            * (triangle_vertices[i, x2] - triangle_vertices[j, x2])
            + a_22
            * (triangle_vertices[m, x1] - triangle_vertices[j, x1])
            * (triangle_vertices[j, x1] - triangle_vertices[i, x1]),
        ],
        [
            a_11
            * (triangle_vertices[j, x2] - triangle_vertices[m, x2])
            * (triangle_vertices[m, x2] - triangle_vertices[i, x2])
            + a_22
            * (triangle_vertices[m, x1] - triangle_vertices[j, x1])
            * (triangle_vertices[i, x1] - triangle_vertices[m, x1]),
            a_11 * (triangle_vertices[m, x2] - triangle_vertices[i, x2]) ** 2
            + a_22 * (triangle_vertices[i, x1] - triangle_vertices[m, x1]) ** 2,
            a_11
            * (triangle_vertices[m, x2] - triangle_vertices[i, x2])
            * (triangle_vertices[i, x2] - triangle_vertices[j, x2])
            + a_22
            * (triangle_vertices[i, x1] - triangle_vertices[m, x1])
            * (triangle_vertices[j, x1] - triangle_vertices[i, x1]),
        ],
        [
            a_11
            * (triangle_vertices[j, x2] - triangle_vertices[m, x2])
            * (triangle_vertices[i, x2] - triangle_vertices[j, x2])
            + a_22
            * (triangle_vertices[m, x1] - triangle_vertices[j, x1])
            * (triangle_vertices[j, x1] - triangle_vertices[i, x1]),
            a_11
            * (triangle_vertices[m, x2] - triangle_vertices[i, x2])
            * (triangle_vertices[i, x2] - triangle_vertices[j, x2])
            + a_22
            * (triangle_vertices[i, x1] - triangle_vertices[m, x1])
            * (triangle_vertices[j, x1] - triangle_vertices[i, x1]),
            a_11 * (triangle_vertices[i, x2] - triangle_vertices[j, x2]) ** 2
            + a_22 * (triangle_vertices[j, x1] - triangle_vertices[i, x1]) ** 2,
        ],
    ]
    area = compute_area(triangle_vertices)
    ke = np.array(ke)
    return (1 / (2 * area)) * ke


def compute_area(triangle_vertices):
    S = 0.5 * (
        (
            triangle_vertices[0, 0] * triangle_vertices[1, 1]
            + triangle_vertices[1, 0] * triangle_vertices[2, 1]
            + triangle_vertices[2, 0] * triangle_vertices[0, 1]
        )
        - (
            triangle_vertices[0, 1] * triangle_vertices[1, 0]
            + triangle_vertices[1, 1] * triangle_vertices[2, 0]
            + triangle_vertices[2, 1] * triangle_vertices[0, 0]
        )
    )
    return 2 * S


def compute_me(triangle_vertices):
    M = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    return (compute_area(triangle_vertices) / 24) * M


def compute_qe(triangle_vertices, fe):
    Me = compute_me(triangle_vertices)
    return np.dot(Me, np.transpose(np.array(fe)))


def assemble_global_matrix(globalMatrix, stiffnessMatrix, triangle):
    for i in range(3):
        for j in range(3):
            globalMatrix[triangle[i], triangle[j]] += stiffnessMatrix[i, j]
    return globalMatrix


def assemble_rhs(Qe, triangle, globalB):
    for i in range(3):
        globalB[triangle[i]] += Qe[i]
    return globalB


def compute_pressure(A, G, c, chi, x, d=2, k=0):
    return k + (G - chi) * c - G - A * G * ((dot_product(x, x)) / (2 * d))


def dot_product(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")

    product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    return product


def compute_gradient(triangle_vertices, concentration):
    x = triangle_vertices[:, 0]
    y = triangle_vertices[:, 1]

    A = np.array([[x[1] - x[0], x[2] - x[0]], [y[1] - y[0], y[2] - y[0]]])

    A_inv = np.linalg.inv(A)

    delta_c = np.array(
        [concentration[1] - concentration[0], concentration[2] - concentration[0]]
    )

    gradient = np.dot(A_inv, delta_c)

    return gradient


### Quadratic approximations
def compute_midpoints(triangle_vertices):
    midpoints = np.zeros((3, 2))
    midpoints[0] = (triangle_vertices[0] + triangle_vertices[1]) / 2
    midpoints[1] = (triangle_vertices[1] + triangle_vertices[2]) / 2
    midpoints[2] = (triangle_vertices[2] + triangle_vertices[0]) / 2
    return midpoints


def compute_ke_quadratic(triangle_vertices, a_11, a_22):
    midpoints = compute_midpoints(triangle_vertices)
    vertices = np.vstack((triangle_vertices, midpoints))

    ke = np.zeros((6, 6))

    for i in range(3):
        for j in range(3):
            ke[i, j] = a_11 * (vertices[(i + 1) % 3, 1] - vertices[(i + 2) % 3, 1]) * (
                vertices[(j + 1) % 3, 1] - vertices[(j + 2) % 3, 1]
            ) + a_22 * (vertices[(i + 2) % 3, 0] - vertices[(i + 1) % 3, 0]) * (
                vertices[(j + 2) % 3, 0] - vertices[(j + 1) % 3, 0]
            )
            ke[i + 3, j + 3] = ke[i, j]

    for i in range(3):
        ke[i, i + 3] = ke[i + 3, i] = 0.5 * (ke[i, (i + 1) % 3] + ke[i, (i + 2) % 3])

    area = compute_area(triangle_vertices)
    return (1 / (2 * area)) * ke


def compute_me_quadratic(triangle_vertices):
    M = np.array(
        [
            [4, 2, 2, 1, 1, 1],
            [2, 4, 2, 1, 1, 1],
            [2, 2, 4, 1, 1, 1],
            [1, 1, 1, 2, 1, 1],
            [1, 1, 1, 1, 2, 1],
            [1, 1, 1, 1, 1, 2],
        ]
    )
    area = compute_area(triangle_vertices)
    return (area / 120) * M


def compute_qe_quadratic(triangle_vertices, fe):
    Me = compute_me_quadratic(triangle_vertices)
    return np.dot(Me, np.transpose(np.array(fe)))


def assemble_global_matrix_quadratic(globalMatrix, stiffnessMatrix, triangle):
    for i in range(6):
        for j in range(6):
            globalMatrix[triangle[i], triangle[j]] += stiffnessMatrix[i, j]
    return globalMatrix


def assemble_rhs_quadratic(Qe, triangle, globalB):
    for i in range(6):
        globalB[triangle[i]] += Qe[i]
    return globalB
