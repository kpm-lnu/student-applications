import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def fem_poisson_2d(n, m):
    # Mesh parameters
    hx = 1.0 / n
    hy = 1.0 / m
    Nx1 = n + 1
    Ny1 = m + 1
    N = Nx1 * Ny1

    # Assemble stiffness matrix
    main_diag = np.ones(N) * 2.0 / hx / hx + 2.0 / hy / hy
    off_diag_x = np.ones(N - 1) * -1.0 / hx / hx
    off_diag_x[np.arange(1, N) % Nx1 == 0] = 0.0
    off_diag_y = np.ones(N - Ny1) * -1.0 / hy / hy
    diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
    K = sp.diags(diagonals, [0, -1, 1, -Ny1, Ny1], format='csr')

    # Assemble force vector
    b = hx * hy * np.ones(N)  # Assumes f(x, y) = 1

    # Solve system of equations
    U = spla.spsolve(K, b)

    # Reshape solution vector to 2D array
    U = U.reshape((Ny1, Nx1))

    return U

def linear_interpolation(x, y, Z, Nx_interp, Ny_interp):
    x_interp = np.linspace(x[0], x[-1], Nx_interp)
    y_interp = np.linspace(y[0], y[-1], Ny_interp)
    X_interp, Y_interp = np.meshgrid(x_interp, y_interp)
    Z_interp = np.zeros((Ny_interp, Nx_interp))

    for i in range(Ny_interp):
        for j in range(Nx_interp):
            xi = X_interp[i, j]
            yi = Y_interp[i, j]

            indices_x = np.where(x <= xi)[0][-2:]  # Індекси найближчих точок по x
            indices_y = np.where(y <= yi)[0][-2:]  # Індекси найближчих точок по y

            if len(indices_x) < 2 or len(indices_y) < 2:
                # Недостатньо точок для лінійної інтерполяції, використовуємо нульове значення замість
                Z_interp[i, j] = 0
            else:
                x1, x2 = x[indices_x]
                y1, y2 = y[indices_y]
                z11 = Z[indices_y[0], indices_x[0]]
                z12 = Z[indices_y[0], indices_x[1]]
                z21 = Z[indices_y[1], indices_x[0]]
                z22 = Z[indices_y[1], indices_x[1]]

                Z_interp[i, j] = ((x2 - xi) * (y2 - yi) * z11 +
                                  (xi - x1) * (y2 - yi) * z21 +
                                  (x2 - xi) * (yi - y1) * z12 +
                                  (xi - x1) * (yi - y1) * z22) / ((x2 - x1) * (y2 - y1))

    return X_interp, Y_interp, Z_interp

def quadratic_interpolation(x, y, Z, Nx_interp, Ny_interp):
    x_interp = np.linspace(x[0], x[-1], Nx_interp)
    y_interp = np.linspace(y[0], y[-1], Ny_interp)
    X_interp, Y_interp = np.meshgrid(x_interp, y_interp)
    Z_interp = np.zeros((Ny_interp, Nx_interp))

    for i in range(Ny_interp):
        for j in range(Nx_interp):
            xi = X_interp[i, j]
            yi = Y_interp[i, j]

            indices_x = np.where(x <= xi)[0][-3:]  # Індекси найближчих точок по x
            indices_y = np.where(y <= yi)[0][-3:]  # Індекси найближчих точок по y

            if len(indices_x) < 3 or len(indices_y) < 3:
                # Недостатньо точок для квадратичної інтерполяції, використовуємо нульове значення замість
                Z_interp[i, j] = 0
            else:
                x1, x2, x3 = x[indices_x]
                y1, y2, y3 = y[indices_y]
                z11 = Z[indices_y[0], indices_x[0]]
                z12 = Z[indices_y[0], indices_x[1]]
                z13 = Z[indices_y[0], indices_x[2]]
                z21 = Z[indices_y[1], indices_x[0]]
                z22 = Z[indices_y[1], indices_x[1]]
                z23 = Z[indices_y[1], indices_x[2]]
                z31 = Z[indices_y[2], indices_x[0]]
                z32 = Z[indices_y[2], indices_x[1]]
                z33 = Z[indices_y[2], indices_x[2]]

                Z_interp[i, j] = ((x2 - xi) * (y2 - yi) * (x3 - xi) * (y3 - yi) * z11 +
                                  (xi - x1) * (y2 - yi) * (x3 - xi) * (y3 - yi) * z21 +
                                  (x2 - xi) * (yi - y1) * (x3 - xi) * (y3 - yi) * z12 +
                                  (xi - x1) * (yi - y1) * (x3 - xi) * (y3 - yi) * z22 +
                                  (x2 - xi) * (y2 - yi) * (xi - x1) * (y3 - yi) * z13 +
                                  (xi - x1) * (y2 - yi) * (xi - x1) * (y3 - yi) * z23 +
                                  (x2 - xi) * (yi - y1) * (xi - x1) * (y3 - yi) * z33 +
                                  (xi - x1) * (yi - y1) * (xi - x1) * (y3 - yi) * z32 +
                                  (x2 - xi) * (y2 - yi) * (x3 - xi) * (yi - y1) * z31 +
                                  (xi - x1) * (y2 - yi) * (x3 - xi) * (yi - y1) * z11 +
                                  (x2 - xi) * (yi - y1) * (x3 - xi) * (yi - y1) * z21 +
                                  (xi - x1) * (yi - y1) * (x3 - xi) * (yi - y1) * z12 +
                                  (x2 - xi) * (y2 - yi) * (xi - x1) * (yi - y1) * z22 +
                                  (xi - x1) * (y2 - yi) * (xi - x1) * (yi - y1) * z32 +
                                  (x2 - xi) * (yi - y1) * (xi - x1) * (yi - y1) * z23 +
                                  (xi - x1) * (yi - y1) * (xi - x1) * (yi - y1) * z33) / (
                                      (x2 - x1) * (y2 - y1) * (x3 - x1) * (y3 - y1))

    return X_interp, Y_interp, Z_interp

def cubic_interpolation(x, y, Z, Nx_interp, Ny_interp):
    x_interp = np.linspace(x[0], x[-1], Nx_interp)
    y_interp = np.linspace(y[0], y[-1], Ny_interp)
    X_interp, Y_interp = np.meshgrid(x_interp, y_interp)
    Z_interp = np.zeros((Ny_interp, Nx_interp))

    for i in range(Ny_interp):
        for j in range(Nx_interp):
            xi = X_interp[i, j]
            yi = Y_interp[i, j]

            indices_x = np.where(x <= xi)[0][-4:]  # Індекси найближчих точок по x
            indices_y = np.where(y <= yi)[0][-4:]  # Індекси найближчих точок по y

            if len(indices_x) < 4 or len(indices_y) < 4:
                # Недостатньо точок для кубічної інтерполяції, використовуємо нульове значення замість
                Z_interp[i, j] = 0
            else:
                x1, x2, x3, x4 = x[indices_x]
                y1, y2, y3, y4 = y[indices_y]
                z11 = Z[indices_y[0], indices_x[0]]
                z12 = Z[indices_y[0], indices_x[1]]
                z13 = Z[indices_y[0], indices_x[2]]
                z14 = Z[indices_y[0], indices_x[3]]
                z21 = Z[indices_y[1], indices_x[0]]
                z22 = Z[indices_y[1], indices_x[1]]
                z23 = Z[indices_y[1], indices_x[2]]
                z24 = Z[indices_y[1], indices_x[3]]
                z31 = Z[indices_y[2], indices_x[0]]
                z32 = Z[indices_y[2], indices_x[1]]
                z33 = Z[indices_y[2], indices_x[2]]
                z34 = Z[indices_y[2], indices_x[3]]
                z41 = Z[indices_y[3], indices_x[0]]
                z42 = Z[indices_y[3], indices_x[1]]
                z43 = Z[indices_y[3], indices_x[2]]
                z44 = Z[indices_y[3], indices_x[3]]

                Z_interp[i, j] = ((x2 - xi) * (y2 - yi) * (x3 - xi) * (y3 - yi) * (x4 - xi) * (y4 - yi) * z11 +
                                  (xi - x1) * (y2 - yi) * (x3 - xi) * (y3 - yi) * (x4 - xi) * (y4 - yi) * z21 +
                                  (x2 - xi) * (yi - y1) * (x3 - xi) * (y3 - yi) * (x4 - xi) * (y4 - yi) * z12 +
                                  (xi - x1) * (yi - y1) * (x3 - xi) * (y3 - yi) * (x4 - xi) * (y4 - yi) * z22 +
                                  (x2 - xi) * (y2 - yi) * (xi - x1) * (y3 - yi) * (x4 - xi) * (y4 - yi) * z13 +
                                  (xi - x1) * (y2 - yi) * (xi - x1) * (y3 - yi) * (x4 - xi) * (y4 - yi) * z23 +
                                  (x2 - xi) * (yi - y1) * (xi - x1) * (y3 - yi) * (x4 - xi) * (y4 - yi) * z33 +
                                  (xi - x1) * (yi - y1) * (xi - x1) * (y3 - yi) * (x4 - xi) * (y4 - yi) * z43 +
                                  (x2 - xi) * (y2 - yi) * (x3 - xi) * (yi - y1) * (x4 - xi) * (y4 - yi) * z14 +
                                  (xi - x1) * (y2 - yi) * (x3 - xi) * (yi - y1) * (x4 - xi) * (y4 - yi) * z24 +
                                  (x2 - xi) * (yi - y1) * (x3 - xi) * (yi - y1) * (x4 - xi) * (y4 - yi) * z34 +
                                  (xi - x1) * (yi - y1) * (x3 - xi) * (yi - y1) * (x4 - xi) * (y4 - yi) * z44 +
                                  (x2 - xi) * (y2 - yi) * (x3 - xi) * (y3 - yi) * (xi - x1) * (y4 - yi) * z41 +
                                  (xi - x1) * (y2 - yi) * (x3 - xi) * (y3 - yi) * (xi - x1) * (y4 - yi) * z11 +
                                  (x2 - xi) * (yi - y1) * (x3 - xi) * (y3 - yi) * (xi - x1) * (y4 - yi) * z21 +
                                  (xi - x1) * (yi - y1) * (x3 - xi) * (y3 - yi) * (xi - x1) * (y4 - yi) * z12 +
                                  (x2 - xi) * (y2 - yi) * (xi - x1) * (y3 - yi) * (xi - x1) * (y4 - yi) * z22 +
                                  (xi - x1) * (y2 - yi) * (xi - x1) * (y3 - yi) * (xi - x1) * (y4 - yi) * z32 +
                                  (x2 - xi) * (yi - y1) * (xi - x1) * (y3 - yi) * (xi - x1) * (y4 - yi) * z42 +
                                  (xi - x1) * (yi - y1) * (xi - x1) * (y3 - yi) * (xi - x1) * (y4 - yi) * z33 +
                                  (x2 - xi) * (y2 - yi) * (x3 - xi) * (yi - y1) * (xi - x1) * (y4 - yi) * z23 +
                                  (xi - x1) * (y2 - yi) * (x3 - xi) * (yi - y1) * (xi - x1) * (y4 - yi) * z33 +
                                  (x2 - xi) * (yi - y1) * (x3 - xi) * (yi - y1) * (xi - x1) * (y4 - yi) * z43 +
                                  (xi - x1) * (yi - y1) * (x3 - xi) * (yi - y1) * (xi - x1) * (y4 - yi) * z33 +
                                  (x2 - xi) * (y2 - yi) * (x3 - xi) * (y3 - yi) * (x4 - xi) * (yi - y1) * z41 +
                                  (xi - x1) * (y2 - yi) * (x3 - xi) * (y3 - yi) * (x4 - xi) * (yi - y1) * z11 +
                                  (x2 - xi) * (yi - y1) * (x3 - xi) * (y3 - yi) * (x4 - xi) * (yi - y1) * z21 +
                                  (xi - x1) * (yi - y1) * (x3 - xi) * (y3 - yi) * (x4 - xi) * (yi - y1) * z12 +
                                  (x2 - xi) * (y2 - yi) * (xi - x1) * (y3 - yi) * (x4 - xi) * (yi - y1) * z22 +
                                  (xi - x1) * (y2 - yi) * (xi - x1) * (y3 - yi) * (x4 - xi) * (yi - y1) * z32 +
                                  (x2 - xi) * (yi - y1) * (xi - x1) * (y3 - yi) * (x4 - xi) * (yi - y1) * z42 +
                                  (xi - x1) * (yi - y1) * (xi - x1) * (y3 - yi) * (x4 - xi) * (yi - y1) * z33 +
                                  (x2 - xi) * (y2 - yi) * (x3 - xi) * (yi - y1) * (x4 - xi) * (yi - y1) * z23 +
                                  (xi - x1) * (y2 - yi) * (x3 - xi) * (yi - y1) * (x4 - xi) * (yi - y1) * z33 +
                                  (x2 - xi) * (yi - y1) * (x3 - xi) * (yi - y1) * (x4 - xi) * (yi - y1) * z43 +
                                  (xi - x1) * (yi - y1) * (x3 - xi) * (yi - y1) * (x4 - xi) * (yi - y1) * z33) / (
                                      (x2 - x1) * (y2 - y1) * (x3 - x1) * (y3 - y1) * (x4 - x1) * (y4 - y1))

    return X_interp, Y_interp, Z_interp

def plot_interpolation_results(x, y, Z, X_interp, Y_interp, Z_interp, method_name):
    # Вивід значень інтерполяції
    print(f"Значення інтерполяції за методом {method_name}:")
    for i in range(len(x)):
        for j in range(len(y)):
            print(f"({X_interp[i, j]:.2f}, {Y_interp[i, j]:.2f}): {Z_interp[i, j]:.4f}")

    # Розрахунок різниці
    difference = np.abs(Z_interp - Z)

    # Побудова графіків
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12))

    im1 = ax1.imshow(Z_interp, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f"Інтерполяція за методом {method_name}")
    fig.colorbar(im1, ax=ax1, label='Значення')

    im2 = ax2.imshow(difference, extent=(0, 1, 0, 1), origin='lower', cmap='coolwarm')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title("Різниця")
    fig.colorbar(im2, ax=ax2, label='Різниця')

    im3 = ax3.imshow(Z, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title("Оригінальна функція")
    fig.colorbar(im3, ax=ax3, label='Значення')



    plt.tight_layout()
    plt.show()


def main():
    # Введення значень Nx та Ny
    print("Введіть кількість точок Nx та Ny:")
    Nx = int(input("Nx: "))
    Ny = int(input("Ny: "))

    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    Z = fem_poisson_2d(Nx, Ny)

    # Визначення значень Nx_interp та Ny_interp
    Nx_interp = Nx + 1
    Ny_interp = Ny + 1

    # Лінійна інтерполяція
    X_interp_linear, Y_interp_linear, Z_interp_linear = linear_interpolation(x, y, Z, Nx_interp, Ny_interp)
    method_name_linear = "Лінійна інтерполяція"
    plot_interpolation_results(x, y, Z, X_interp_linear, Y_interp_linear, Z_interp_linear, method_name_linear)

    # Квадратична інтерполяція
    X_interp_quadratic, Y_interp_quadratic, Z_interp_quadratic = quadratic_interpolation(x, y, Z, Nx_interp, Ny_interp)
    method_name_quadratic = "Квадратична інтерполяція"
    plot_interpolation_results(x, y, Z, X_interp_quadratic, Y_interp_quadratic, Z_interp_quadratic, method_name_quadratic)

    # Кубічна інтерполяція
    X_interp_cubic, Y_interp_cubic, Z_interp_cubic = cubic_interpolation(x, y, Z, Nx_interp, Ny_interp)
    method_name_cubic = "Кубічна інтерполяція"
    plot_interpolation_results(x, y, Z, X_interp_cubic, Y_interp_cubic, Z_interp_cubic, method_name_cubic)

main()
