import numpy as np
import time

def csr_matvec(values, col_idx, row_ptr, x):
    """Множення матриці у форматі CSR на вектор x."""
    n = len(row_ptr) - 1
    result = np.zeros(n)
    for i in range(n):
        for j in range(row_ptr[i], row_ptr[i + 1]):
            result[i] += values[j] * x[col_idx[j]]
    return result

def conjugate_gradient_csr(values, col_idx, row_ptr, b, tol=1e-10):
    """Метод спряжених градієнтів для матриці у форматі CSR."""
    n = len(b)
    x = np.zeros(n)
    r = b - csr_matvec(values, col_idx, row_ptr, x)
    p = r.copy()
    rs_old = np.dot(r, r)
    iterations = 0

    for _ in range(n):
        iterations += 1
        Ap = csr_matvec(values, col_idx, row_ptr, p)
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, iterations

def load_crs_matrix(filename):
    """Завантажує CRS-матрицю з файлу .npz."""
    data = np.load(filename)
    return data['values'], data['col_idx'], data['row_ptr']

def calculate_crs_memory(values, col_idx, row_ptr):
    """Обчислює обсяг пам’яті, використаної для зберігання матриці у форматі CRS (у MB)."""
    memory_values = values.nbytes
    memory_col_idx = col_idx.nbytes
    memory_row_ptr = row_ptr.nbytes
    total_memory = (memory_values + memory_col_idx + memory_row_ptr) / (1024 * 1024)  # У MB
    return total_memory

def compare_methods_with_crs(matrix_files):
    """Порівнює метод спряжених градієнтів для заданих матриць у форматі CRS і виводить табличний результат."""
    
    print(f"{'Matrix Size':<15}{'Iterations':<12}{'Time (sec)':<15}{'Memory (MB)':<15}")
    print("=" * 70)
    
    for filename in matrix_files:
        size = filename.split('_')[2]  # Витягуємо розмір матриці з назви файлу
        values, col_idx, row_ptr = load_crs_matrix(filename + ".npz")
        b = np.random.rand(len(row_ptr) - 1)  # Генеруємо випадковий вектор b для розв’язання

        # Обчислюємо використану пам’ять
        memory_usage = calculate_crs_memory(values, col_idx, row_ptr)

        # Виконуємо метод спряжених градієнтів
        start_time = time.time()
        _, iterations = conjugate_gradient_csr(values, col_idx, row_ptr, b)
        elapsed_time = time.time() - start_time
        
        # Форматований табличний вивід з пам’яттю
        print(f"{size:<15}{iterations:<12}{elapsed_time:<15.3f}{memory_usage:<15.2f}")
    
    print("=" * 70)

if __name__ == "__main__":
    # Імена файлів із збереженими матрицями CRS
    matrix_files = ["crs_matrix_100x100", "crs_matrix_250x250", "crs_matrix_500x500"]
    compare_methods_with_crs(matrix_files)
