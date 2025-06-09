import numpy as np
from generate_slar import generate_slar

def to_crs(A):
    """Конвертує матрицю A у формат Compressed Row Storage (CRS)."""
    values, col_idx, row_ptr = [], [], [0]
    
    for row in A:
        for j, value in enumerate(row):
            if value != 0:
                values.append(value)
                col_idx.append(j)
        row_ptr.append(len(values))
    
    return np.array(values), np.array(col_idx), np.array(row_ptr)

def save_crs_matrix(filename, values, col_idx, row_ptr):
    """Зберігає CRS-матрицю у файл .npz."""
    np.savez(filename, values=values, col_idx=col_idx, row_ptr=row_ptr)
    print(f"Матриця збережена у форматі CRS у файлі {filename}.npz")

if __name__ == "__main__":
    matrix_sizes = [100, 250, 500]  # Розміри матриць для генерації
    for size in matrix_sizes:
        print(f"\n=== Генеруємо та конвертуємо матрицю розміром {size} x {size} у CRS ===")
        A, _ = generate_slar(size, density=0.1)  # Генеруємо матрицю зі щільністю 10%
        values, col_idx, row_ptr = to_crs(A)  # Перетворюємо у CRS
        filename = f"crs_matrix_{size}x{size}"  # Ім'я файлу для збереження
        save_crs_matrix(filename, values, col_idx, row_ptr)  # Зберігаємо матрицю
