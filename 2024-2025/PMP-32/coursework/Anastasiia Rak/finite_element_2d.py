from enum import Enum


def apply_boundary_conditions(matrix, f_load, p, m, nodes, ug, ap=1):
    validate_boundary_conditions(ug)
    bounds = get_boundary_elements_and_nodes(p, m, ug, ap)
    n = len(ug)
    for i in range(n):
        for index in bounds[i][1]:  # [1] — список вузлів
            for j in range(len(matrix[index])):
                matrix[index][j] = 0
            matrix[index][index] = 1
            f_load[index] = ug[i][1](nodes[index][0], nodes[index][1])  # <== ось тут виправлено
    return (matrix, f_load)


def get_boundary_elements_and_nodes(p, m, ug, degree=1):
    bounds = []
    # gamma_i consists of [elements], [nodes]
    elements_1 = [j for j in range(p)]
    nodes_1 = [j for j in range(1, degree * p)]

    elements_2 = [p * (i + 1) - 1 for i in range(m)]
    nodes_2 = [(i + 1) * (degree * p + 1) - 1 for i in range(1, degree * m)]

    intersection = degree * p
    # print(intersection)
    if (ug[0][0] == TypeOfBoundCond.DIRICHLET):
        nodes_1.append(intersection)
        # print('1', nodes_1)
    else:
        nodes_2.insert(0, intersection)
        # print('2', nodes_2)

    elements_3 = [p * m - 1 - j for j in range(p)]
    nodes_3 = [(degree * p + 1) * degree * m + j for j in reversed(range(1, degree * p))]

    intersection = (degree * p + 1) * (degree * m + 1) - 1
    # print(intersection)
    if (ug[1][0] == TypeOfBoundCond.DIRICHLET):
        nodes_2.append(intersection)
        # print('2', nodes_2)
    else:
        nodes_3.insert(0, intersection)
        # print('3', nodes_3)

    elements_4 = [p * i for i in reversed(range(m))]
    nodes_4 = [i * (degree * p + 1) for i in reversed(range(1, degree * m))]

    intersection = degree * m * (degree * p + 1)
    # print(intersection)
    if (ug[2][0] == TypeOfBoundCond.DIRICHLET):
        nodes_3.append(intersection)
        # print('3', nodes_3)
    else:
        nodes_4.insert(0, intersection)
        # print('4', nodes_4)

    intersection = 0
    # print(intersection)
    if (ug[3][0] == TypeOfBoundCond.DIRICHLET):
        nodes_4.append(intersection)
        # print('4', nodes_4)
    else:
        nodes_1.insert(0, intersection)
        # print('1', nodes_1)

    gamma_1 = [elements_1, nodes_1]
    bounds.append(gamma_1)
    gamma_2 = [elements_2, nodes_2]
    bounds.append(gamma_2)
    gamma_3 = [elements_3, nodes_3]
    bounds.append(gamma_3)
    gamma_4 = [elements_4, nodes_4]
    bounds.append(gamma_4)
    return bounds


class TypeOfBoundCond(Enum):
    DIRICHLET = 1
    NEUMANN = 2


def validate_boundary_conditions(ug):
    isPresentDirichlet = False
    for ug_ in ug:
        if ug_[0] == TypeOfBoundCond.DIRICHLET:
            isPresentDirichlet = True
            break
    if not isPresentDirichlet:
        raise Exception("Wrong boundary conditions: all are Neumann, Dirichlet must be present")


def get_boundary_points(p, m, ap=1):
    bounds = []
    gamma_1 = [];
    gamma_2 = [];
    gamma_3 = [];
    gamma_4 = []
    for j in range(ap * p + 1):
        gamma_1.append(j)
    bounds.append(gamma_1)
    for i in range(1, ap * m + 1):
        index = (i + 1) * (ap * p + 1) - 1
        gamma_2.append(index)
    bounds.append(gamma_2)
    for j in reversed(range(ap * p)):
        index = (ap * p + 1) * ap * m + j
        gamma_3.append(index)
    bounds.append(gamma_3)
    for i in reversed(range(1, ap * m)):
        index = i * (ap * p + 1)
        gamma_4.append(index)
    bounds.append(gamma_4)
    return bounds
