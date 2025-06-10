import numpy as np

def integrand(x):
    return np.sin(x)  # Функція для інтегрування, можна замінити на іншу

# Метод трапецій для чисельного інтегрування
def trapezoidal_rule(f, a, b, n):
    x = np.linspace(a, b, n)
    y = f(x)
    h = (b - a) / (n - 1)  # Крок
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral

# Функція для визначення оптимальної кількості вузлів
def find_optimal_nodes(f, a, b, exact_value, tolerance=1e-6, max_nodes=1000):
    n = 2  # Початкова кількість вузлів (мінімальна)
    previous_integral = 0
    while n <= max_nodes:
        integral = trapezoidal_rule(f, a, b, n)
        error = np.abs(integral - exact_value)  # Погрішність
        if error <= tolerance:
            return n, integral, error  # Повертаємо кількість вузлів, результат інтеграції та помилку
        n += 1  # Збільшуємо кількість вузлів для наступної ітерації
    return n, integral, error  # Якщо максимальна кількість вузлів досягнута
