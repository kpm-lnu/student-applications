import numpy as np
import sympy as sp
import mesh_2d as m2d
import system_2d as s2d
import finite_element_2d as f2e
import graph2d as g2d


def ug_1(x, y):
    return 0
    # return y**2
    # return y
    # return y**2


def ug_2(x, y):
    return 0
    # return y**2
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


def k1(x, y):
    return 1


def k2(x, y):
    return 1


# test1
def ft1(x, y, A=100, x0=0.5, y0=0.5, const=0.1):
    return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / const)


def ft2(x, y, A=100, x0=0.5, y0=0.5, const=1):
    return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / const)


# test2
sourcest3 = [(0.5, 0.5)]
strengthst3 = [100]

sourcest4 = [(0.5, 0.5), (0.2, 0.8)]
strengthst4 = [100, 50]

atol = 0.05

ft3 = lambda x, y: sum(
    s for (x0, y0), s in zip(sourcest3, strengthst3)
    if np.isclose(x, x0, atol=atol) and np.isclose(y, y0, atol=atol)
)

ft4 = lambda x, y: sum(
    s for (x0, y0), s in zip(sourcest4, strengthst4)
    if np.isclose(x, x0, atol=atol) and np.isclose(y, y0, atol=atol)
)


# test3
def k1t5(x, y):
    return 0.1 if 0.45 < x < 1.55 else 10.0


def k2t5(x, y):
    return 1.0


def ft5(x, y):
    return 100 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.001)


def k1t6(x, y):
    if 0.45 < x < 1.55:
        return 0.05
    return 1.0


def k2t6(x, y):
    if 0.65 < y < 1.75:
        return 0.01
    return 1.0


def ft6(x, y):
    if x < 0.3 and y < 0.3:
        return 20
    elif x > 0.7 and y > 0.7:
        return 5
    else:
        return 0


# ver
# def fv(x, y):
#     return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
def fv(x, y):
    return 2 * sp.pi ** 2 * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)


def exact_solution(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def main():
    verticest1 = [(0, 0), (1.5, 0), (1, 1), (0, 0.75)]
    verticest2 = [(0, 0), (1.5, 0), (1, 1), (0, 0.75)]
    verticest3 = [(0, 0), (1, 0), (1.1, 1), (0, 0.9)]
    verticest4 = [(0, 0), (1, 0), (1.1, 1), (0, 0.9)]
    verticest5 = [(0, 0), (1.5, 0), (1, 1), (0, 0.75)]
    verticest6 = [(0, 0), (1, 0), (1, 1), (0, 1)]
    verticesv = [(0, 0), (1, 0), (1, 1), (0, 1)]

    ap = 1

    ug = [
        [f2e.TypeOfBoundCond.DIRICHLET, ug_3],  # нижнє (y = 0)
        [f2e.TypeOfBoundCond.DIRICHLET, ug_2],  # ліве (x = 0)
        [f2e.TypeOfBoundCond.DIRICHLET, ug_4],  # праве (x = 1)
        [f2e.TypeOfBoundCond.DIRICHLET, ug_1]  # верхнє (y = 1)
    ]

    element_type = 'D2QU4N'
    p = 10
    m = 10

    # test1
    # NLt1, ELt1 = m2d.uniform_mesh_with_vertices(verticest1, p, m, element_type, ap)
    # f_loadt1 = s2d.set_up_vector(ft1, NLt1, ELt1, p, m, ap)
    # elem_matricest1 = s2d.compute_element_stiffness(ELt1, NLt1, ap, k1=k1, k2=k2)
    # matrixt1 = s2d.assemble_global_stiffness_matrix(NLt1, ELt1, p, m, elem_matricest1, ap)
    # matrixt1, f_loadt1 = f2e.apply_boundary_conditions(matrixt1, f_loadt1, p, m, NLt1, ug, ap)
    # ut1 = np.linalg.solve(matrixt1, f_loadt1)
    # g2d.plot_2d_solution(ut1, NLt1, ELt1)
    # print("Координати вузлів:\n", NLt1)
    # print("Елементи:\n", ELt1)
    # print(f_loadt1)
    # print(matrixt1)
    # print(ut1)

    # test2
    # NLt2, ELt2= m2d.uniform_mesh_with_vertices(verticest2, p, m, element_type, ap)
    # f_loadt2 = s2d.set_up_vector(ft2, NLt2, ELt2, p, m, ap)
    # elem_matricest2 = s2d.compute_element_stiffness(ELt2, NLt2, ap, k1=k1, k2=k2)
    # matrixt2 =  s2d.assemble_global_stiffness_matrix(NLt2, ELt2, p, m, elem_matricest2, ap)
    # matrixt2, f_loadt2 = f2e.apply_boundary_conditions(matrixt2, f_loadt2, p, m, NLt2, ug, ap)
    # ut2 = np.linalg.solve(matrixt2, f_loadt2)
    # g2d.plot_2d_solution(ut2, NLt2, ELt2)
    # print("Координати вузлів:\n", NLt2)
    # print("Елементи:\n", ELt2)
    # print(f_loadt2)
    # print(matrixt2)
    # print(ut2)

    # test3
    # NLt3, ELt3= m2d.uniform_mesh_with_vertices(verticest3, p, m, element_type, ap)
    # f_loadt3 = s2d.set_up_vector_point_sources(sourcest3, strengthst3, NLt3, p, m, ap)
    # elem_matricest3 = s2d.compute_element_stiffness(ELt3, NLt3, ap, k1=k1, k2=k2)
    # matrixt3 =  s2d.assemble_global_stiffness_matrix(NLt3, ELt3, p, m, elem_matricest3, ap)
    # matrixt3, f_loadt3 = f2e.apply_boundary_conditions(matrixt3, f_loadt3, p, m, NLt3, ug, ap)
    # ut3 = np.linalg.solve(matrixt3, f_loadt3)
    # g2d.plot_2d_solution(ut3, NLt3, ELt3)
    # print("Координати вузлів:\n", NLt3)
    # print("Елементи:\n", ELt3)
    # print(f_loadt3)
    # print(matrixt3)
    # print(ut3)

    # test4
    # NLt4, ELt4= m2d.uniform_mesh_with_vertices(verticest4, p, m, element_type, ap)
    # f_loadt4 = s2d.set_up_vector_point_sources(sourcest4, strengthst4, NLt4, p, m, ap)
    # elem_matricest4 = s2d.compute_element_stiffness(ELt4, NLt4, ap, k1=k1, k2=k2)
    # matrixt4 =  s2d.assemble_global_stiffness_matrix(NLt4, ELt4, p, m, elem_matricest4, ap)
    # matrixt4, f_loadt4 = f2e.apply_boundary_conditions(matrixt4, f_loadt4, p, m, NLt4, ug, ap)
    # ut4 = np.linalg.solve(matrixt4, f_loadt4)
    # g2d.plot_2d_solution(ut4, NLt4, ELt4)
    # print("Координати вузлів:\n", NLt4)
    # print("Елементи:\n", ELt4)
    # print(f_loadt4)
    # print(matrixt4)
    # print(ut4)

    # test5
    # NLt5, ELt5= m2d.uniform_mesh_with_vertices(verticest5, p, m, element_type, ap)
    # f_loadt5 = s2d.set_up_vector(ft5, NLt5, ELt5, p, m, ap)
    # elem_matricest5 = s2d.compute_element_stiffness(ELt5, NLt5, ap, k1=k1t5, k2=k2t5)
    # matrixt5 =  s2d.assemble_global_stiffness_matrix(NLt5, ELt5, p, m, elem_matricest5, ap)
    # matrixt5, f_loadt5 = f2e.apply_boundary_conditions(matrixt5, f_loadt5, p, m, NLt5, ug, ap)
    # ut5 = np.linalg.solve(matrixt5, f_loadt5)
    # g2d.plot_2d_solution(ut5, NLt5, ELt5)
    # print("Координати вузлів:\n", NLt5)
    # print("Елементи:\n", ELt5)
    # print(f_loadt5)
    # print(matrixt5)
    # print(ut5)

    # test6
    # NLt6, ELt6= m2d.uniform_mesh_with_vertices(verticest6, p, m, element_type, ap)
    # f_loadt6 = s2d.set_up_vector(ft6, NLt6, ELt6, p, m, ap)
    # elem_matricest6 = s2d.compute_element_stiffness(ELt6, NLt6, ap, k1=k1t6, k2=k2t6)
    # matrixt6 =  s2d.assemble_global_stiffness_matrix(NLt6, ELt6, p, m, elem_matricest6, ap)
    # matrixt6, f_loadt6 = f2e.apply_boundary_conditions(matrixt6, f_loadt6, p, m, NLt6, ug, ap)
    # ut6 = np.linalg.solve(matrixt6, f_loadt6)
    # g2d.plot_2d_solution(ut6, NLt6, ELt6)
    # print("Координати вузлів:\n", NLt6)
    # print("Елементи:\n", ELt6)
    # print(f_loadt6)
    # print(matrixt6)
    # print(ut6)

    # ver
    NLv, ELv = m2d.uniform_mesh_with_vertices(verticesv, p, m, element_type, ap)
    f_loadv = s2d.set_up_vector(fv, NLv, ELv, p, m, ap)
    elem_matricesv = s2d.compute_element_stiffness(ELv, NLv, ap, k1=k1, k2=k2)
    matrixv = s2d.assemble_global_stiffness_matrix(NLv, ELv, p, m, elem_matricesv, ap)
    matrixv, f_loadv = f2e.apply_boundary_conditions(matrixv, f_loadv, p, m, NLv, ug, ap)
    uv = np.linalg.solve(matrixv, f_loadv)
    g2d.plot_2d_solution(uv, NLv, ELv)
    print("Координати вузлів:\n", NLv)
    print("Елементи:\n", ELv)
    print(f_loadv)
    print(matrixv)
    print(uv)
    g2d.plot_2d_solution2(uv, NLv, ELv, exact_solution=exact_solution)
    g2d.plot_2d_solution_exact(exact_solution, NLv)

    # g2d.plot_2d_solution_difference(uv, NLv, exact_solution)


if __name__ == '__main__':
    main()
