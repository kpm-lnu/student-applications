import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import dblquad
import scipy.integrate as scin
from mpl_toolkits.mplot3d import Axes3D


ksi_left = -1
ksi_right = 1
eta_left = -1
eta_right = 1

def f(x, y):
    #return (1 + x**2)*(1 + 2*y**2)
    return 1

def uniform_mesh(d1, d2, p, m, element_type):
    PD = 2  # Простір (x, y)
    q = np.array([[0, 0], [d1, 0], [0, d2], [d1, d2]])  # 4 кути прямокутника

    NoN = (p + 1) * (m + 1)  # Кількість вузлів
    NoE = p * m  # Кількість елементів (для прямокутників)
    NPE = 4 if element_type == 'D2QU4N' else 3  # Кількість вузлів на елемент

    NL = np.zeros([NoN, PD])
    a = (q[1, 0] - q[0, 0]) / p  # Інкременти по x
    b = (q[2, 1] - q[0, 1]) / m  # Інкременти по y

    n = 0
    for i in range(m + 1):
        for j in range(p + 1):
            NL[n, 0] = q[0, 0] + j * a  # Значення x
            NL[n, 1] = q[0, 1] + i * b  # Значення y
            n += 1

    if element_type == 'D2QU4N':  # Прямокутники
        EL = np.zeros([NoE, NPE], dtype=int)
        for i in range(m):
            for j in range(p):
                n1 = i * (p + 1) + j
                n2 = n1 + 1
                n3 = n1 + (p + 1)
                n4 = n3 + 1
                EL[i * p + j] = [n1, n2, n4, n3]
    elif element_type == 'D2TR3N':  # Трикутники
        EL = np.zeros([NoE * 2, NPE], dtype=int)  # Подвійна кількість елементів для трикутників
        e = 0
        for i in range(m):
            for j in range(p):
                n1 = i * (p + 1) + j
                n2 = n1 + 1
                n3 = n1 + (p + 1)
                n4 = n3 + 1
                # Перший трикутник
                EL[e] = [n1, n2, n4]
                e += 1
                # Другий трикутник
                EL[e] = [n1, n4, n3]
                e += 1
    else:
        raise ValueError("Неправильний тип елемента. Використовуйте 'D2QU4N' або 'D2TR3N'.")

    return NL, EL

def get_base_functions(m=1):
    # Повертає базисні функції для m+1 x m+1 вузлів
    n = (m + 1) * (m + 1)  # Кількість вузлів
    base_functions = [0 for _ in range(n)]
    ksi, eta = sp.symbols('ksi eta')

    # Координати вузлів у локальній системі
    h_ksi = (ksi_right - ksi_left) / m
    h_eta = (eta_right - eta_left) / m

    submesh = [[ksi_left + j * h_ksi, eta_left + i * h_eta]
               for i in range(m + 1) for j in range(m + 1)]

    for i in range(n):
        expression = 1
        for k in range(n):
            if k != i:
                dx = submesh[i][0] - submesh[k][0]
                dy = submesh[i][1] - submesh[k][1]
                if dx != 0:
                    expression *= (ksi - submesh[k][0]) / dx
                if dy != 0:
                    expression *= (eta - submesh[k][1]) / dy
        base_functions[i] = expression  # Без спрощення на цьому етапі

    return base_functions


def N(i, ksi, eta, ap):
    """Базисні функції для прямокутних елементів різного порядку."""
    if ap == 1:
        # Лінійні базисні функції
        if i == 0:
            return (1 - ksi) * (1 - eta) / 4
        elif i == 1:
            return (1 + ksi) * (1 - eta) / 4
        elif i == 2:
            return (1 + ksi) * (1 + eta) / 4
        elif i == 3:
            return (1 - ksi) * (1 + eta) / 4
    elif ap == 2:
        # Квадратичні базисні функції
        if i < 4:
            return (1 + ksi * (-1)**(i // 2)) * (1 + eta * (-1)**(i % 2)) / 4
        elif i == 4:
            return (1 - ksi**2) * (1 - eta) / 2
        elif i == 5:
            return (1 + ksi) * (1 - eta**2) / 2
        elif i == 6:
            return (1 - ksi**2) * (1 + eta) / 2
        elif i == 7:
            return (1 - ksi) * (1 - eta**2) / 2
        elif i == 8:
            return (1 - ksi**2) * (1 - eta**2)
    elif ap == 3:
        # Кубічні базисні функції
        if i < 4:
            # Вершини
            return (1 + ksi * (-1)**(i // 2)) * (1 + eta * (-1)**(i % 2)) / 4
        elif i < 12:
            # Середні вузли
            if i in [4, 5, 6, 7]:  # Горизонтальні сторони
                return (1 - ksi**2) * (1 + eta * (-1)**(i % 2)) / 4
            else:  # Вертикальні сторони
                return (1 + ksi * (-1)**((i - 8) % 2)) * (1 - eta**2) / 4
        else:
            # Внутрішні вузли
            return (1 - ksi**2) * (1 - eta**2)
    else:
        raise ValueError("Unsupported approximation order.")
    return 0



def calculate_b(NL, EL, p, m, ap):
    """Обчислення вектора правої частини для заданої сітки та апроксимації."""
    # Кількість вузлів на елемент залежить від ап.
    NPE = 4 if ap == 1 else (9 if ap == 2 else 16)
    b = np.zeros(NL.shape[0])  # Ініціалізація глобального вектора правої частини

    # Інтегрування для кожного елемента
    for e in range(EL.shape[0]):
        nodes = EL[e, :NPE]  # Вузли елемента для відповідного порядку
        x_coords = NL[nodes, 0]
        y_coords = NL[nodes, 1]

        # Вирази для перетворення координат
        def x_expr(ksi, eta):
            return sum(N(i, ksi, eta, ap) * x_coords[i] for i in range(NPE))

        def y_expr(ksi, eta):
            return sum(N(i, ksi, eta, ap) * y_coords[i] for i in range(NPE))

        def det_jacobian(ksi, eta):
            """Обчислення детермінанта Якобіана для елемента."""
            # Похідні базисних функцій за ksi та eta
            dN_dksi = [
                (1 / 4) * (-(1 - eta) if i == 0 else (1 - eta) if i == 1 else (1 + eta) if i == 2 else -(1 + eta))
                for i in range(NPE)
            ]
            dN_deta = [
                (1 / 4) * (-(1 - ksi) if i == 0 else -(1 + ksi) if i == 1 else (1 + ksi) if i == 2 else (1 - ksi))
                for i in range(NPE)
            ]

            # Елементи матриці Якобіана
            J11 = sum(dN_dksi[i] * x_coords[i] for i in range(NPE))
            J12 = sum(dN_dksi[i] * y_coords[i] for i in range(NPE))
            J21 = sum(dN_deta[i] * x_coords[i] for i in range(NPE))
            J22 = sum(dN_deta[i] * y_coords[i] for i in range(NPE))

            # Обчислення детермінанта Якобіана
            return abs(J11 * J22 - J12 * J21)

        # Інтегральна функція для вузла i
        def integrand(ksi, eta, i):
            x = x_expr(ksi, eta)
            y = y_expr(ksi, eta)
            return N(i, ksi, eta, ap) * f(x, y) * det_jacobian(ksi, eta)

        # Обчислення локальних інтегралів для кожного вузла елемента
        for i in range(NPE):
            integral, _ = dblquad(
                lambda eta, ksi: integrand(ksi, eta, i),
                -1, 1,  # Межі для ksi
                lambda ksi: -1, lambda ksi: 1  # Межі для eta
            )
            b[nodes[i]] += integral
    print("x_expr:", x_expr(0, 0), "y_expr:", y_expr(0, 0))
    print("det_jacobian:", det_jacobian(0, 0))
    print("N(0, 0, 0, 1):", N(0, 0, 0, 1))

    return b


from scipy.interpolate import griddata

def plot_2d_solution(u, NL, EL, exact_solution=None):
    """
    Побудова графіка наближеного розв'язку в 2D-просторі із прямокутною сіткою.

    u - знайдений вектор розв'язку (значення на вузлах).
    NL - координати вузлів (матриця розмірності [NoN x 2]).
    EL - елементи сітки (матриця розмірності [NoE x NPE]).
    exact_solution - функція для точного розв'язку (опціонально).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Координати вузлів
    x = NL[:, 0]
    y = NL[:, 1]
    z = u

    # Створення регулярної прямокутної сітки
    x_lin = np.linspace(np.min(x), np.max(x), 100)
    y_lin = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(x_lin, y_lin)

    # Інтерполяція значень u для регулярної сітки
    Z = griddata((x, y), z, (X, Y), method='linear')

    # Побудова поверхні
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none', label='Наближений розв\'язок')

    # Якщо заданий точний розв'язок
    if exact_solution:
        Z_exact = exact_solution(X, Y)
        ax.plot_surface(X, Y, Z_exact, cmap='plasma', alpha=0.4, label='Точний розв\'язок')

    # Налаштування графіка
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    ax.set_title('Наближений розв\'язок у 2D-просторі')
    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    plt.show()

def uniform_mesh(d1, d2, p, m, element_type, degree=1, b1=0, b2=0):
    """
    Генерація сітки вузлів і елементів для різних рівнів точності.

    d1, d2 - межі по x і y.
    p, m - кількість елементів по x і y.
    element_type - тип елементу ('D2QU4N' або інші).
    degree - рівень деталізації (1, 2 або 3).
    b1, b2 - початок координат (за замовчуванням 0, 0).

    Повертає:
    NL - матриця вузлів.
    EL - матриця елементів.
    """
    PD = 2  # Розмірність простору (x, y)
    nodes_x = degree * p + 1  # Кількість вузлів по x
    nodes_y = degree * m + 1  # Кількість вузлів по y

    # Генерація вузлів
    NL = np.zeros([nodes_x * nodes_y, PD])
    x_coords = np.linspace(b1, d1, nodes_x)
    y_coords = np.linspace(b2, d2, nodes_y)

    n = 0
    for y in y_coords:
        for x in x_coords:
            NL[n, :] = [x, y]
            n += 1

    # Визначення кількості вузлів на елемент
    if degree == 1:
        NPE = 4
    elif degree == 2:
        NPE = 9
    elif degree == 3:
        NPE = 16
    else:
        raise ValueError("Параметр degree повинен бути 1, 2 або 3.")

    NoE = p * m  # Кількість елементів
    EL = np.zeros([NoE, NPE], dtype=int)

    # Генерація елементів
    e = 0
    for i in range(m):
        for j in range(p):
            base = i * degree * nodes_x + j * degree
            if degree == 1:
                # 4 вузли на елемент
                n1 = base
                n2 = n1 + 1
                n3 = n1 + nodes_x
                n4 = n3 + 1
                EL[e, :] = [n1, n2, n3, n4]
            elif degree == 2:
                # 9 вузлів на елемент
                n1 = base
                n2 = n1 + 2
                n3 = n1 + 2 * nodes_x
                n4 = n3 + 2
                n5 = n1 + 1
                n6 = n3 + 1
                n7 = n1 + nodes_x
                n8 = n7 + 2
                n9 = n7 + 1
                EL[e, :] = [n1, n5, n2, n7, n9, n8, n3, n6, n4]
            elif degree == 3:
                # 16 вузлів на елемент
                n1 = base
                n2 = n1 + 3
                n3 = n1 + 3 * nodes_x
                n4 = n3 + 3
                n5 = n1 + 1
                n6 = n1 + 2
                n7 = n3 + 1
                n8 = n3 + 2
                n9 = n1 + nodes_x
                n10 = n9 + 1
                n11 = n9 + 2
                n12 = n9 + 3
                n13 = n1 + 2 * nodes_x
                n14 = n13 + 1
                n15 = n13 + 2
                n16 = n13 + 3
                EL[e, :] = [n1, n5, n6, n2, n9, n10, n11, n12, n13, n14, n15, n16, n3, n7, n8, n4]
            e += 1

    return NL, EL


def uniform_mesh_level1(d1, d2, p, m, element_type, b1=0, b2=0):
    PD = 2  # Простір (x, y)
    q = np.array([[b1, b2], [d1, b2], [b1, d2], [d1, d2]])  # 4 кути прямокутника

    NoN = (p + 1) * (m + 1)  # Кількість вузлів
    NoE = p * m  # Кількість елементів (для прямокутників)
    NPE = 4 if element_type == 'D2QU4N' else 3  # Кількість вузлів на елемент

    NL = np.zeros([NoN, PD])
    a = (q[1, 0] - q[0, 0]) / p  # Інкременти по x
    b = (q[2, 1] - q[0, 1]) / m  # Інкременти по y

    n = 0
    for i in range(m + 1):
        for j in range(p + 1):
            NL[n, 0] = q[0, 0] + j * a  # Значення x
            NL[n, 1] = q[0, 1] + i * b  # Значення y
            n += 1

    EL = np.zeros([NoE, NPE], dtype=int)
    for i in range(m):
        for j in range(p):
            n1 = i * (p + 1) + j
            n2 = n1 + 1
            n3 = n1 + (p + 1)
            n4 = n3 + 1
            EL[i * p + j] = [n1, n2, n3, n4]

    return NL, EL

def uniform_mesh_level2(d1, d2, p, m, element_type, b1=0, b2=0):
    PD = 2  # Простір (x, y)
    nodes_x = 2 * p + 1  # Подвоєна кількість вузлів по x (основні + середні)
    nodes_y = 2 * m + 1  # Подвоєна кількість вузлів по y (основні + середні)

    NL = np.zeros([nodes_x * nodes_y, PD])
    x_coords = np.linspace(b1, d1, nodes_x)  # Координати вузлів по x
    y_coords = np.linspace(b2, d2, nodes_y)  # Координати вузлів по y

    n = 0
    for y in y_coords:
        for x in x_coords:
            NL[n, :] = [x, y]
            n += 1

    NoE = p * m  # Кількість елементів
    EL = np.zeros([NoE, 9], dtype=int)  # 8 вузлів на елемент (лише на краях)

    e = 0
    for i in range(m):
        for j in range(p):
            n1 = i * 2 * nodes_x + j * 2
            n2 = n1 + 2
            n3 = n1 + 2 * nodes_x
            n4 = n3 + 2
            n5 = n1 + 1
            n6 = n3 + 1
            n7 = n1 + nodes_x
            n8 = n7 + 2
            n9 = n7 + 1
            # [n1, n5, n2, n8, n4, n6, n3, n7, n9]
            EL[e, :] = [n1, n5, n2, n7, n9, n8, n3, n6, n4]
            e += 1

    return NL, EL


def uniform_mesh_level3(d1, d2, p, m, element_type,b1=0, b2=0):
    PD = 2  # Простір (x, y)
    nodes_x = 3 * p + 1  # Потроєна кількість вузлів по x
    nodes_y = 3 * m + 1  # Потроєна кількість вузлів по y

    NL = np.zeros([nodes_x * nodes_y, PD])
    x_coords = np.linspace(b1, d1, nodes_x)  # Координати вузлів по x
    y_coords = np.linspace(b2, d2, nodes_y)  # Координати вузлів по y

    n = 0
    for y in y_coords:
        for x in x_coords:
            NL[n, :] = [x, y]
            n += 1

    NoE = p * m  # Кількість елементів
    EL = np.zeros([NoE, 16], dtype=int)  # 16 вузлів на елемент

    e = 0
    for i in range(m):
        for j in range(p):
            n1 = i * 3 * nodes_x + j * 3
            n2 = n1 + 3
            n3 = n1 + 3 * nodes_x
            n4 = n3 + 3
            n5 = n1 + 1
            n6 = n1 + 2
            n7 = n3 + 1
            n8 = n3 + 2
            n9 = n1 + nodes_x
            n10 = n9 + 1
            n11 = n9 + 2
            n12 = n9 + 3
            n13 = n1 + 2 * nodes_x
            n14 = n13 + 1
            n15 = n13 + 2
            n16 = n13 + 3
            EL[e, :] = [n1, n5, n6, n2, n9, n10, n11, n12, n13, n14, n15, n16, n3, n7, n8, n4]
            e += 1

    return NL, EL


def main():
    d1 = 2  # Кінцева координата по x
    d2 = 3  # Кінцева координата по y
    b1 = 1  # Початкова координата по x
    b2 = 2  # Початкова координата по y
    p = 2  # Поділ по x
    m = 2  # Поділ по y

    # Рівень 1
    NL1, EL1 = uniform_mesh_level1(d1, d2, p, m, 'D2QU4N', b1, b2)
    print(calculate_b(NL1, EL1, p, m, 1))

    # Рівень 2
    NL2, EL2 = uniform_mesh_level2(d1, d2, p, m, 'D2QU4N', b1, b2)
    print(calculate_b(NL2, EL2, p, m, 2))

    # Рівень 3
    NL3, EL3 = uniform_mesh_level3(d1, d2, p, m, 'D2QU4N', b1, b2)
    print(calculate_b(NL3, EL3, p, m, 3))






if __name__ == "__main__":
    main()
