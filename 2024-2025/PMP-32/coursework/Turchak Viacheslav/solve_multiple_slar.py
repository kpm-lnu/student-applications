import numpy as np
import time
import sys

def conjugate_gradient_procedural(A, b, tol=1e-10):
    """Розв’язання СЛАР методом спряжених градієнтів (процедурний підхід)."""
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)
    iterations = 0

    for _ in range(n):
        iterations += 1
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, iterations

def calculate_memory_usage(A):
    """Обчислює обсяг пам’яті, використаної для зберігання матриці в повному форматі (MB)."""
    full_memory = A.nbytes / (1024 * 1024)  # Пам’ять у MB
    return full_memory

def solve_and_analyze(matrix_sizes, value_range):
    """Формує таблицю результатів для заданих розмірів матриць."""
    print(f"{'Size':<10}{'Iterations':<12}{'Time (sec)':<12}{'Memory (MB)':<15}")
    print("-" * 60)

    for size in matrix_sizes:
        # Генеруємо матрицю з розширеним діапазоном значень
        A = np.random.uniform(-value_range, value_range, (size, size))
        A = (A + A.T) / 2  # Робимо матрицю симетричною
        A += size * np.eye(size)  # Додаємо size * I для забезпечення позитивної визначеності
        b = np.random.uniform(-value_range, value_range, size)  # Генеруємо вектор b з тим же діапазоном

        start_time = time.time()
        x, iterations = conjugate_gradient_procedural(A, b)  # Розв'язуємо СЛАР
        elapsed_time = time.time() - start_time

        # Обчислюємо використану пам’ять
        full_memory = calculate_memory_usage(A)

        # Виводимо результати у форматованій таблиці
        print(f"{size:<10}{iterations:<12}{elapsed_time:<12.3f}{full_memory:<15.2f}")

if __name__ == "__main__":
    matrix_sizes = [100, 250, 500]  # Розміри матриць
    value_range = 1000  # Діапазон значень матриці від -1000 до 1000
    solve_and_analyze(matrix_sizes, value_range)  # Викликаємо аналіз
