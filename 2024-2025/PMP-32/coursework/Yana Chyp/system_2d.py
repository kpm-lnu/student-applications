# from distutils.command.install import value
import cmath
from enum import Enum

import numpy as np

import mesh_2d as m2d
import base_functions_2d as bf2d
import finite_element_2d as fe2d
import base_functions_2d as bs2d
from base_functions_2d import ksi_right, ksi_left, eta_right, eta_left
import sympy as sp
import scipy.integrate as scin

def solve(b1, d1, b2, d2, p, m, degree, f, ug, K = [1, 1], element_type='D2QU4N'):
    u, nodes, elements = get_solution(b1, d1, b2, d2, p, m, degree, f, ug, K, element_type)
    m2d.plot_2d_solution(u, nodes, elements)

def get_solution(b1, d1, b2, d2, p, m, degree, f, ug, K = [1, 1], element_type='D2QU4N'):
    nodes, elements = m2d.uniform_mesh(d1, d2, p, m, element_type, degree, b1, b2)
    h_x = (d1 - b1) / p
    h_y = (d2 - b2) / m
    J = h_x * h_y / 4
    #J = J / (4 * p * m)
    # J = J * 4 / cmath.sqrt(p * m);
    matrix = set_up_matrix(nodes, elements, degree, J, p, m, K)
    base = bs2d.get_base_functions(degree)
    vec_of_integrals = fe2d.integrate_base_functions(base, degree)
    # print('vec of integrals:')
    # print('[' + ', '.join([f"{el:.4f}" for el in vec_of_integrals]) + ']')

    f_vec = set_up_vector(f, base, nodes, elements, degree, h_x, h_y, p, m)
    f_vec = [v*J for v in f_vec]

    # print('f_vec: ')
    # print('[' + ', '.join([f"{el:.4f}" for el in f_vec]) + ']')
    # print('matrix: ')
    # for row in matrix:
    #     print('[' + ', '.join([f"{el:.3f}" for el in row]) + ']' )
    matrix, f_vec = apply_boundary_conditions(matrix, f_vec, p, m, J, nodes, ug, elements, vec_of_integrals, degree)
    # print('f_vec: ')
    # print('[' + ', '.join([f"{el:.4f}" for el in f_vec]) + ']')

    u = np.linalg.solve(matrix, f_vec)
    # print('solution: ')
    # print('[' + ', '.join([f"{el:.4f}" for el in u]) + ']')
    return u, nodes, elements


def set_up_matrix(nodes, elements, degree, J, p, m, K = [1, 1]):
    # nodes, elements = m2d.uniform_mesh(d1, d2, p, m, element_type, degree)
    #npe - nodes per element (assume all are the same)
    npe = len(elements[0])
    #let it be rectangle

    #assume nodes are equidistant
    # h_x = (d1-b1)/p /degree; h_y = (d2-b2)/m /degree; J = h_x*h_y/((ksi_right-ksi_left)*(eta_right-eta_left))

    n = (degree*p+1)*(degree*m+1)
    matrix = [[0 for j in range(n)] for i in range(n)]

    em = fe2d.element_matrix(degree, K)
    #ioe - index of element
    for ioe in range(len(elements)):
        #noe - nodes of element
        noe = elements[ioe]
        # print(noe)
        for i in range(len(noe)):
            for j in range(len(noe)):
                im = noe[i]
                jm = noe[j]
                value = em[i][j]*J
                # print("(", i, ",  ", j, ") -> (", im, ", ", jm, "); value = ", f"{value:.4f}")
                matrix[im][jm] += value
            # print('[' + ', '.join([f"{el:.8f}" for el in matrix[i]]) + ']')
    return matrix


def set_up_vector(f, base, nodes, elements, degree, h_x, h_y, p, m):
    # nodes, elements = m2d.uniform_mesh(d1, d2, p, m, element_type, degree)

    n = (degree*p+1)*(m*degree+1)
    # h_x = (d1-b1) / p; h_y = (d2-b2) / m
    J = h_x*h_y/4 #/ p / m / 4
    f_vec = [0 for i in range(n)]
    ksi = sp.symbols('ksi'); eta = sp.symbols('eta')
    x = sp.symbols('x'); y = sp.symbols('y')

    for ioe in range(len(elements)):
        noe = elements[ioe]
        x_0 = nodes[noe[0]][0]; y_0 = nodes[noe[0]][1]
        # lin_ksi = 2*(x - x_0)/h_x - 1
        # lin_eta = 2*(y - y_0)/h_y - 1
        lin_x = (ksi+1)*h_x/2 + x_0
        lin_y = (eta+1)*h_y/2 + y_0
        f_expr = sp.sympify(f(x, y)).subs('x', lin_x).subs('y', lin_y)

        # print(f_expr)
        for i in range(len(noe)):
            N_i = base[i]   #.subs('ksi', lin_ksi).subs('eta', lin_eta)
            prod = N_i*f_expr
            # print(prod)
            prod = sp.lambdify([ksi, eta], prod)
            value = scin.dblquad(prod, -1, 1, -1, 1)[0]
            # print(ioe, x_0, x_0+h_x, y_0, y_0+h_y, value*J)
            f_vec[noe[i]] += value*J
    return f_vec




def get_boundary_elements_and_nodes(p, m, ug, degree = 1):
    bounds = []
    #gamma_i consists of [elements], [nodes]
    elements_1 = [j for j in range(p)]
    nodes_1 = [j for j in range(1, degree*p)]

    elements_2 = [p*(i+1)-1 for i in range(m)]
    nodes_2 = [(i+1)*(degree*p+1)-1 for i in range(1, degree*m)]

    intersection = degree*p
    # print(intersection)
    if (ug[0][0]==TypeOfBoundCond.DIRICHLET):
        nodes_1.append(intersection)
        # print('1', nodes_1)
    else:
        nodes_2.insert(0, intersection)
        # print('2', nodes_2)

    elements_3 = [p*m-1-j for j in range(p)]
    nodes_3 = [(degree*p+1)*degree*m + j for j in reversed(range(1, degree*p))]

    intersection = (degree*p+1)*(degree*m+1)-1
    # print(intersection)
    if (ug[1][0]==TypeOfBoundCond.DIRICHLET):
        nodes_2.append(intersection)
        # print('2', nodes_2)
    else:
        nodes_3.insert(0, intersection)
        # print('3', nodes_3)

    elements_4 = [p*i for i in reversed(range(m))]
    nodes_4 = [i*(degree*p+1) for i in reversed(range(1, degree*m))]

    intersection = degree*m*(degree*p+1)
    # print(intersection)
    if (ug[2][0]==TypeOfBoundCond.DIRICHLET):
        nodes_3.append(intersection)
        # print('3', nodes_3)
    else:
        nodes_4.insert(0, intersection)
        # print('4', nodes_4)

    intersection = 0
    # print(intersection)
    if (ug[3][0]==TypeOfBoundCond.DIRICHLET):
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

# ug = [[TYPE, ug_3], [TYPE, ug_2], [TYPE, ug_4], [TYPE, ug_1]]

def validate_boundary_conditions(ug):
    isPresentDirichlet = False
    for ug_ in ug:
        if ug_[0] == TypeOfBoundCond.DIRICHLET:
            isPresentDirichlet = True
            break
    if not isPresentDirichlet:
        raise Exception("Wrong boundary conditions: all are Neumann, Dirichlet must be present")


def apply_boundary_conditions(matrix, f_vec, p, m, J, nodes, ug, elements, vec_of_integrals, degree):
    validate_boundary_conditions(ug)
    bounds = get_boundary_elements_and_nodes(p, m, ug, degree)
    # print(bounds)
    for i in range(len(ug)):
        if ug[i][0]==TypeOfBoundCond.DIRICHLET:
            apply_dirichlet(matrix, f_vec, nodes, bounds[i], ug[i][1])
        # if ug[i][0]==TypeOfBoundCond.NEUMANN:
        else:
            apply_neumann(f_vec, vec_of_integrals, J, bounds[i], ug[i][1], elements, degree)
    f_vec = np.float64(f_vec)
    return (matrix, f_vec)

def apply_dirichlet(matrix, f_vec, nodes, bound, ug_func_i):
    for index in bound[1]:
        for j in range(len(matrix[index])):
            matrix[index][j] = 0
        matrix[index][index] = 1
        f_vec[index] = ug_func_i(nodes[index][0], nodes[index][1])

def apply_neumann(f_vec, vec_of_integrals, J, bound, ug_func, elements, degree):
    # print(bound[0], bound[1])
    # for index in bound[1]:
    #     for j in range(len(matrix[index])):
    #         matrix[index][j] = 0
    #     matrix[index][index] = 1
    for i in range(len(bound[0])):
            #ioc - index of current node
        for ioc in range(len(bound[1])):
            #assume ug_func is a constant
            #else must be put inside integral
            # print(ioc, ug_func, ug_func(1,1))
            # print('before: ', f_vec[bound[1][ioc]])
            f_vec[bound[1][ioc]] += ug_func(1, 1) * J * get_integral_for_node(bound[0][i], bound[1][ioc], vec_of_integrals, elements)
            # print('after: ', f_vec[bound[1][ioc]])

def index_in_element(el_num, node_num, elements):
    index = np.where(elements[el_num] == node_num)[0]
    if index.size==0:
        return -1
    return index[0]

def get_integral_for_node(el_num, node_num, vec_of_integrals, elements):
    index = index_in_element(el_num, node_num, elements)
    if index < 0:
        return 0
    # print(el_num, node_num, index)
    #responds to the base function
    return vec_of_integrals[index]