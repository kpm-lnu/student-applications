import numpy as np
import matplotlib.pyplot as plt
import base_functions_2d as b2f
import finite_element_2d as f2e
import mesh_2d as m2d
from scipy.interpolate import griddata


def plot_statistics(start, end, statistics, u_exact, n_points=100):
    abscissas = np.linspace(0, 1, n_points)
    xs = [(1 - t) * start[0] + t * end[0] for t in abscissas]
    ys = [(1 - t) * start[1] + t * end[1] for t in abscissas]
    u_values = [u_exact(x, y) for x, y in zip(xs, ys)]

    plt.plot(abscissas, u_values, color='red', label='u exact')

    for i, (x_approx, u_approx) in enumerate(statistics):
        plt.plot(x_approx, u_approx, label=f'approx {i}')

    plt.xlabel('normalized distance along line')
    plt.ylabel('u')
    plt.title('Exact vs Approx Solutions')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_errors(errors):
    plt.plot([i for i in range(len(errors))], errors, color='green')
    plt.xlabel('degree of nodes')
    plt.ylabel('errors')
    plt.grid(True)
    plt.show()


def interpolate_solution(x, y, u, NL, EL, ap):
    from scipy.optimize import root

    num_nodes = 4 if ap == 1 else (9 if ap == 2 else 16)

    for element in EL:
        nodes_coords = [NL[i] for i in element]
        x_coords = [pt[0] for pt in nodes_coords]
        y_coords = [pt[1] for pt in nodes_coords]

        def equations(p):
            ksi, eta = p
            x_mapped = sum(b2f.N(i, ksi, eta, ap) * x_coords[i] for i in range(num_nodes))
            y_mapped = sum(b2f.N(i, ksi, eta, ap) * y_coords[i] for i in range(num_nodes))
            return [x_mapped - x, y_mapped - y]

        sol = root(equations, [0, 0])

        if sol.success:
            ksi, eta = sol.x
            if -1 <= ksi <= 1 and -1 <= eta <= 1:
                u_local = [u[i] for i in element]
                u_val = sum(u_local[i] * b2f.N(i, ksi, eta, ap) for i in range(num_nodes))
                return u_val

    return 0  # якщо точка не належить жодному елементу


def plot_2d_solution(u, NL, EL, exact_solution=None):
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


def calculate_L2_error(u_exact, u_values, nodes, elements, base, degree):
    from numpy.polynomial.legendre import leggauss
    dN_dksi_list, dN_deta_list = f2e.compute_partial_derivatives(degree)
    nq = degree + 1
    quad_points, quad_weights = leggauss(nq)

    error_squared = 0

    for element in elements:
        element_coords = [nodes[i] for i in element]
        u_local = [u_values[i] for i in element]

        for i in range(nq):
            for j in range(nq):
                ξ = quad_points[i]
                η = quad_points[j]
                w = quad_weights[i] * quad_weights[j]

                x = sum(base[k](ξ, η) * element_coords[k][0] for k in range(len(element)))
                y = sum(base[k](ξ, η) * element_coords[k][1] for k in range(len(element)))

                u_h_val = sum(u_local[k] * base[k](ξ, η) for k in range(len(element)))
                u_ex_val = u_exact(x, y)

                J_val = m2d.compute_jacobian(ξ, η, [coord[0] for coord in element_coords],
                                             [coord[1] for coord in element_coords], dN_dksi_list, dN_deta_list)
                error_squared += (u_ex_val - u_h_val) ** 2 * w * J_val
    return np.sqrt(error_squared)


def plot_2d_solution2(u, NL, EL, exact_solution=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = NL[:, 0]
    y = NL[:, 1]
    z = u

    x_lin = np.linspace(np.min(x), np.max(x), 100)
    y_lin = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = griddata((x, y), z, (X, Y), method='linear')

    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none', label='Наближений розв\'язок')

    if exact_solution:
        Z_exact = exact_solution(X, Y)
        Z_computed = griddata((x, y), z, (X, Y), method='linear')

        ax.plot_surface(X, Y, Z_exact, cmap='plasma', alpha=0.4, edgecolor='none')

        mask = ~np.isnan(Z_computed) & ~np.isnan(Z_exact)
        Zc = Z_computed[mask]
        Ze = Z_exact[mask]

        l2_error = np.sqrt(np.mean((Zc - Ze) ** 2))
        max_error = np.max(np.abs(Zc - Ze))

        ax.text2D(0.02, 0.95, f"L2 похибка: {l2_error:.2e}", transform=ax.transAxes, fontsize=12)
        ax.text2D(0.02, 0.91, f"Max похибка: {max_error:.2e}", transform=ax.transAxes, fontsize=12)
        ax.text2D(0.02, 0.87, "Фіолетовий – точний,\nЗелений – FEM", transform=ax.transAxes, fontsize=10)
        print(f"L2 похибка: {l2_error:.2e}")
        print(f"Max похибка: {max_error:.2e}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    ax.set_title('Наближене vs Точне розв\'язання')

    # Додати кольорову шкалу
    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    plt.show()


def plot_2d_solution_exact(exact_solution, NL):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = NL[:, 0]
    y = NL[:, 1]

    x_lin = np.linspace(np.min(x), np.max(x), 100)
    y_lin = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z_exact = exact_solution(X, Y)

    ax.plot_surface(X, Y, Z_exact, cmap='plasma', alpha=0.8, edgecolor='none', label='Точний розв\'язок')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    ax.set_title('Точний розв\'язок')

    plt.colorbar(ax.plot_surface(X, Y, Z_exact, cmap='plasma', alpha=0.8, edgecolor='none'))
    plt.show()


def plot_2d_solution_difference(u, NL, exact_solution=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = NL[:, 0]
    y = NL[:, 1]

    x_lin = np.linspace(np.min(x), np.max(x), 100)
    y_lin = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(x_lin, y_lin)

    Z_exact = exact_solution(X, Y)

    Z_approx = griddata((x, y), u, (X, Y), method='linear')

    difference = Z_approx - Z_exact

    surface = ax.plot_surface(X, Y, difference, cmap='coolwarm', alpha=0.8, edgecolor='none',
                              label='Різниця між точним і наближеним')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Різниця')
    ax.set_title('Різниця між точним і наближеним розв\'язком')

    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    plt.show()
