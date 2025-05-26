import cmath

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

import base_functions_2d as bs2d
import finite_element_2d as fe2d
import system_2d as s2d
import graph2d as g2d
import mesh_2d as m2d


# def exact_solution(x, y):
#     return x*x
def g_1(x, y):
    return y==b2
def g_2(x, y):
    return y==d2
def g_3(x, y):
    return x==b1
def g_4(x, y):
    return x==d1

def u(x, y):
    # return x
    return y**2

    # return sp.sin(sp.pi*x) * sp.sin(sp.pi*y)
def f(x, y):
    # return 0
    # return 1
    # return -2
    # return x
    #return 100 * cmath.e **(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.01)
    return 2* sp.pi**2 * sp.sin(sp.pi*x) * sp.sin(sp.pi*y)
def ug_1(x, y):
    return 0
    # return 1
    # return y**2

def ug_2(x, y):
    # return 1
    # return (-(y-0.5)**2 + 0.25)*8 + 1
    # return y**2
    return 0

def ug_3(x, y):
    #return - sp.pi * sp.sin(sp.pi*x) * sp.sin(sp.pi*y)
    # return x
    return 0
    # return x*(1-x)
    # return y
def ug_4(x, y):
    #return - sp.pi * sp.sin(sp.pi*x) * sp.sin(sp.pi*y)
    # return x
    # return 1
    # return x*(1-x)
    return 0
    # return y
b1 = 0
b2 = 0
d1 = 1
d2 = 1
p = 4
m = 4
degree = 3
K = [1, 1]
element_type='D2QU4N'
ug = [[s2d.TypeOfBoundCond.DIRICHLET, ug_3],
      [s2d.TypeOfBoundCond.DIRICHLET, ug_2],
      [s2d.TypeOfBoundCond.DIRICHLET, ug_4],
      [s2d.TypeOfBoundCond.DIRICHLET, ug_1]]

s2d.solve(b1, d1, b2, d2, p, m, degree, f, ug, K, element_type)

# base = bs2d.get_base_functions(degree)
# for f in base:
#     print(f)

# em = fe2d.element_matrix(degree)
# for row in em:
#     print('[' + ', '.join([f"{el:.3f}" for el in row]) + ']' )
#



# if degree == 1:
#     nodes, elements = m2d.uniform_mesh_level1(d1, d2, p, m, element_type)
# elif degree == 2:
#     nodes, elements = m2d.uniform_mesh_level2(d1, d2, p, m, element_type)
# elif degree == 3:
#     nodes, elements = m2d.uniform_mesh_level3(d1, d2, p, m, element_type)
# else:
#     print("unsupported degree")

# for i in range(len(nodes)): print(nodes[i])

# bounds = s2d.get_boundary_points(p, m)
# print(bounds)



# matrix = s2d.set_up_matrix(d1, d2, p, m, element_type, degree)
# print(50*"-")
# for row in matrix:
#     print('[' + ', '.join([f"{el:.4f}" for el in row]) + ']' )

# print(50*"-")
# f_vec = s2d.set_up_vector(f, base, d1, d2, p, m, element_type, degree)
# print('[' + ', '.join([f"{el:.4f}" for el in f_vec]) + ']')

# matrix, f_vec = s2d.apply_boundary_conditions(matrix, f_vec, p, m, nodes, ug, degree)

# print(50*"-")
# print(50*"-")

# for row in matrix:
#     print('[' + ', '.join([f"{el:.4f}" for el in row]) + ']' )
# print(50*"-")

# print('[' + ', '.join([f"{el:.4f}" for el in f_vec]) + ']')

# u = np.linalg.solve(matrix, f_vec)
# print('[' + ', '.join([f"{el:.4f}" for el in u]) + ']')

# m2d.plot_2d_solution(u, nodes, elements)
