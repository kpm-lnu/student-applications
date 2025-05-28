import numpy as np
import time
from generate_slar import generate_slar

def conjugate_gradient_procedural(A, b, tol=1e-10):
    """Процедурний метод спряжених градієнтів."""
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

def conjugate_gradient_functional(A, b, tol=1e-10):
    """Функціональна реалізація методу спряжених градієнтів."""
    def cg_step(A, x, r, p, rs_old):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = np.dot(r_new, r_new)
        p_new = r_new + (rs_new / rs_old) * p if rs_new > tol else p
        return x_new, r_new, p_new, rs_new

    n = len(b)
    x, r, p = np.zeros(n), b.copy(), b.copy()
    rs_old = np.dot(r, r)
    iterations = 0

    for _ in range(n):
        iterations += 1
        x, r, p, rs_old = cg_step(A, x, r, p, rs_old)
        if np.sqrt(rs_old) < tol:
            break

    return x, iterations

def calculate_memory_usage(A):
    """Обчислює обсяг пам’яті, використаної для зберігання матриці в повному форматі (MB)."""
    full_memory = A.nbytes / (1024 * 1024)  # Пам’ять у MB
    return full_memory

def compare_methods(matrix_sizes):
    """Порівнює методи на заданих розмірах матриць і виводить результати у вигляді таблиці."""
    print(f"{'Method':<15}{'Matrix Size':<15}{'Iterations':<12}{'Time (sec)':<12}{'Memory (MB)':<12}")
    print("-" * 70)

    for size in matrix_sizes:
        A, b = generate_slar(size, density=0.1)  # Генеруємо розріджену матрицю розміром size x size
        
        # Обчислюємо використану пам’ять
        memory_usage = calculate_memory_usage(A)

        # Procedural method
        start_time = time.time()
        _, iterations_proc = conjugate_gradient_procedural(A, b)
        time_proc = time.time() - start_time
        print(f"{'Procedural':<15}{size:<15}{iterations_proc:<12}{time_proc:<12.3f}{memory_usage:<12.2f}")

        # Functional method
        start_time = time.time()
        _, iterations_func = conjugate_gradient_functional(A, b)
        time_func = time.time() - start_time
        print(f"{'Functional':<15}{size:<15}{iterations_func:<12}{time_func:<12.3f}{memory_usage:<12.2f}")

if __name__ == "__main__":
    matrix_sizes = [100, 250, 500]
    compare_methods(matrix_sizes)  # Порівнюємо методи для заданих розмірів матриць
