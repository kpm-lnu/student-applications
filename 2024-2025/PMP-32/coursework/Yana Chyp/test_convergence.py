import cmath
import sympy as sp
import system_2d as s2d
import convergence as con

def u(x, y):
    #return y**2
    # return x**2+y
    # return y**2 + x
    # return x**2 - y**2
    return sp.sin(sp.pi*x) * sp.sin(sp.pi*y)

def f(x, y):
    #return -2
    return 2* sp.pi**2 * sp.sin(sp.pi*x) * sp.sin(sp.pi*y)

def ug_1(x, y):
    return 0
    # return y
    # return y**2
def ug_2(x, y):
    return 0
    # return y+1
    # return y**2 + 1
def ug_3(x, y):
    return 0
    # return x**2
    # return x
def ug_4(x, y):
    return 0
    # return x**2+1
    # return x+1

b1 = 0
b2 = 0
d1 = 1
d2 = 1
# p = 5
# m = 5
degree = 1
K = [1, 1]
element_type='D2QU4N'
ug = [[s2d.TypeOfBoundCond.DIRICHLET, ug_3],
      [s2d.TypeOfBoundCond.DIRICHLET, ug_2],
      [s2d.TypeOfBoundCond.DIRICHLET, ug_4],
      [s2d.TypeOfBoundCond.DIRICHLET, ug_1]]
number = 5
at_x = 0.5
at_y = 0.5

print('evaluating...')
stat, errors = con.get_statistics_at_x(b1, d1, b2, d2, degree, f, ug, at_x, K, element_type, number, u)
print('plotting statistics...')
con.plot_statistics(b2, d2, at_x, stat, u)
print('plotting errors...')
# print(errors)
con.plot_errors(errors)
