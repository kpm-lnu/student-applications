import numpy as np

def generate_slar(rows, cols=None, density=0.5, sparse=True):
    """Генерує матрицю A та вектор b з можливістю вибору її розрідженості."""
    if cols is None:
        cols = rows  # Якщо cols не вказано, робимо квадратну матрицю
    
    A = np.random.rand(rows, cols) * 1000  # Генеруємо випадкові значення в діапазоні [0, 1000]
    A = (A + A.T) / 2  # Робимо її симетричною
    A += rows * np.eye(rows)  # Додаємо rows*I для позитивної визначеності
    
    if sparse:  # Якщо обрано розріджену матрицю
        A[np.random.rand(rows, cols) > density] = 0  # Обнуляємо деякі елементи для розрідження
    
    b = np.random.rand(rows) * 100  # Генеруємо випадковий вектор b у діапазоні [0, 100]
    return np.round(A, 2), np.round(b, 2)  # Округлюємо значення до 2 знаків після коми

if __name__ == "__main__":
    matrix_sizes = [100, 250, 500]  # Розміри матриць, які ми генеруємо
    for size in matrix_sizes:
        print(f"\n=== Генеруємо матрицю розміром {size} x {size} ===")
        A, b = generate_slar(size, density=0.1)  # Генеруємо матрицю зі щільністю ненульових елементів 10%
        print("Матриця A:")
        print(A)
        print("Вектор b:")
        print(b)
