import base_functions_2d as bf2d
import sympy as sp
import scipy.integrate as scin


def element_matrix(m = 1, K = [1, 1]):
    n = (m+1)*(m+1)
    base = bf2d.get_base_functions(m)
    gradients = [[sp.diff(base[k], 'ksi'), sp.diff(base[k], 'eta')]  for k in range(n)]
    # print(gradients)

    #element matrix. set up of integrals of scalar products of gradients
    # em = [[[0,0] for j in range(n)] for i in range(n)]
    em = [[0 for j in range(n)] for i in range(n)]

    # print("products")
    ksi = sp.symbols('ksi')
    eta = sp.symbols('eta')
    for i in range(n):
        for j in range(n):
            prods = (gradients[i][0]*gradients[j][0], gradients[i][1]*gradients[j][1])
            funcs = (sp.lambdify([ksi, eta], prods[0]), sp.lambdify([ksi, eta], prods[1]))
            int_1 = scin.dblquad(funcs[0], bf2d.ksi_left, bf2d.ksi_right, bf2d.eta_left, bf2d.eta_right)[0]
            int_2 = scin.dblquad(funcs[1], bf2d.ksi_left, bf2d.ksi_right, bf2d.eta_left, bf2d.eta_right)[0]
            em[i][j] = K[0]*int_1 + K[1]*int_2
    return em

def integrate_base_functions(base_functions, degree = 1):
    vec = []
    t = sp.symbols('t')
    t_0, t_1, s, ksi_through_t, eta_through_t = bf2d.convert_ksieta_to_t(degree)
    for base_func in base_functions:
        func = bf2d.substitute_t_to_base_func(base_func, ksi_through_t, eta_through_t)
        # print(func)
        vec.append(s*sp.integrate(func, (t, t_0, t_1)))

    return vec