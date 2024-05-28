import numpy as np

a_11 = 1
a_22 = 1


def compute_ke(triangle_vertices, a11, a22):
    i, j, m = 0, 1, 2
    x1, x2 = 0, 1

    ke = [
        [
            a_11 * (triangle_vertices[j, x2] - triangle_vertices[m, x2]) ** 2
            + a_22 * (triangle_vertices[m, x1] - triangle_vertices[j, x1]) ** 2,
            a_11 * (triangle_vertices[j, x2] - triangle_vertices[m, x2]) * (
                        triangle_vertices[m, x2] - triangle_vertices[i, x2])
            + a_22 * (triangle_vertices[m, x1] - triangle_vertices[j, x1]) * (
                        triangle_vertices[i, x1] - triangle_vertices[m, x1]),
            a_11 * (triangle_vertices[j, x2] - triangle_vertices[m, x2]) * (
                        triangle_vertices[i, x2] - triangle_vertices[j, x2])
            + a_22 * (triangle_vertices[m, x1] - triangle_vertices[j, x1]) * (
                        triangle_vertices[j, x1] - triangle_vertices[i, x1])
        ],
        [
            a_11 * (triangle_vertices[j, x2] - triangle_vertices[m, x2]) * (
                        triangle_vertices[m, x2] - triangle_vertices[i, x2])
            + a_22 * (triangle_vertices[m, x1] - triangle_vertices[j, x1]) * (
                        triangle_vertices[i, x1] - triangle_vertices[m, x1]),
            a_11 * (triangle_vertices[m, x2] - triangle_vertices[i, x2]) ** 2
            + a_22 * (triangle_vertices[i, x1] - triangle_vertices[m, x1]) ** 2,
            a_11 * (triangle_vertices[m, x2] - triangle_vertices[i, x2]) * (
                        triangle_vertices[i, x2] - triangle_vertices[j, x2])
            + a_22 * (triangle_vertices[i, x1] - triangle_vertices[m, x1]) * (
                        triangle_vertices[j, x1] - triangle_vertices[i, x1])
        ],
        [
            a_11 * (triangle_vertices[j, x2] - triangle_vertices[m, x2]) * (
                        triangle_vertices[i, x2] - triangle_vertices[j, x2])
            + a_22 * (triangle_vertices[m, x1] - triangle_vertices[j, x1]) * (
                        triangle_vertices[j, x1] - triangle_vertices[i, x1]),
            a_11 * (triangle_vertices[m, x2] - triangle_vertices[i, x2]) * (
                        triangle_vertices[i, x2] - triangle_vertices[j, x2])
            + a_22 * (triangle_vertices[i, x1] - triangle_vertices[m, x1]) * (
                        triangle_vertices[j, x1] - triangle_vertices[i, x1]),
            a_11 * (triangle_vertices[i, x2] - triangle_vertices[j, x2]) ** 2
            + a_22 * (triangle_vertices[j, x1] - triangle_vertices[i, x1]) ** 2
        ]
    ]

    area = compute_area(triangle_vertices)
    ke = np.array(ke)
    return (1 / (2 * area)) * ke


def compute_area(triangle_vertices):
    S = 0.5 * (
            (triangle_vertices[0, 0] * triangle_vertices[1, 1] + triangle_vertices[1, 0] * triangle_vertices[2, 1] +
             triangle_vertices[2, 0] * triangle_vertices[0, 1])
            - (triangle_vertices[0, 1] * triangle_vertices[1, 0] + triangle_vertices[1, 1] * triangle_vertices[2, 0] +
               triangle_vertices[2, 1] * triangle_vertices[0, 0])
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


def exact(x):
    return 0.5 * 3 * x * (3 - x)


def L2_Error(solution, tSolution):
    return np.sqrt(np.sum((solution - tSolution) ** 2))
