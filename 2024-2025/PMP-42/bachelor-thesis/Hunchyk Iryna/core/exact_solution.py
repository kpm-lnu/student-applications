import numpy as np


def exact_solution(x, y):
    """
    Аналітичний розв’язок задачі.
    Підходить до задачі:
        -Δu = f у Ω
        u = 0 на ∂Ω
    де u(x, y) = sin(πx) * sin(πy)
    """
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def rhs_function(x, y):
    """
    Права частина f(x, y), відповідна до точного розв’язку:
        u(x, y) = sin(πx) * sin(πy)
    тоді f(x, y) = 2π² * sin(πx) * sin(πy)
    """
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def exact_solution_vector(nodes):
    """
    Обчислення точного розв’язку у вузлах сітки.

    :param nodes: список координат [(x₀, y₀), (x₁, y₁), ...]
    :return: numpy-масив u_exact[i] = u(x_i, y_i)
    """
    return np.array([exact_solution(x, y) for x, y in nodes])


def rhs_vector(nodes):
    """
    Обчислення правої частини f(x, y) у вузлах сітки.

    :param nodes: список координат [(x₀, y₀), (x₁, y₁), ...]
    :return: numpy-масив f[i] = f(x_i, y_i)
    """
    return np.array([rhs_function(x, y) for x, y in nodes])
