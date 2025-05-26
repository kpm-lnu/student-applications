import sympy as sp


ksi_left = -1
ksi_right = 1
eta_left = -1
eta_right = 1

# def convert_xy_to_ksieta(i, elements, nodes, x, y):
    # ksi =
    # eta =
    # return (ksi, eta)

def convert_ksieta_to_t(degree = 1, ksi_0 = -1, ksi_1 = 1, eta_0 = -1, eta_1 = 1):
    t_0 = 0; t_1 = 1
    s = ((ksi_1-ksi_0)**2 + (eta_1-eta_0)**2)**(1/2)
    t = sp.symbols('t')
    # ksi = sp.symbols('ksi')
    # eta = sp.symbols('eta')
    return t_0, t_1, s, t*(ksi_1-ksi_0)+ksi_0, t*(eta_1-eta_0)+eta_0

def substitute_t_to_base_func(base_func, ksi_through_t, eta_through_t):
    return base_func.subs('ksi', ksi_through_t).subs('eta', eta_through_t)

def get_base_functions(m = 1):
    n = (m+1)*(m+1)     #number of nodes
    base_functions = [0 for k in range(n)]
    ksi = sp.symbols('ksi')
    eta = sp.symbols('eta')

    h_ksi = (ksi_right - ksi_left)/m
    h_eta = (eta_right - eta_left)/m

    submesh = [[ksi_left + k*h_ksi, eta_left + l*h_eta] for l in range(m+1) for k in range(m+1)]
    # print(submesh)
    for i in range(n):
        expression = 1
        for k in range(m+1):
            if submesh[k][0] != submesh[i][0]:
                expression *= ( (ksi - submesh[k][0])/(submesh[i][0]-submesh[k][0]) )
        for l in range(m+1):
            if  submesh[l*(m+1)][1] != submesh[i][1]:
                expression *= ( (eta - submesh[l*(m+1)][1])/(submesh[i][1]-submesh[l*(m+1)][1]) )
        base_functions[i] = sp.simplify(expression)
    return base_functions
