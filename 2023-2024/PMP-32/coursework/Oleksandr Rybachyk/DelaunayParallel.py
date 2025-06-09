import numpy as np
from scipy.spatial import Delaunay
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time

def compute_delaunay(points_chunk):
    return Delaunay(points_chunk)

def merge_triangulations(triangulations, points_chunks):
    # Розбиваємо процес об'єднання на кроки, щоб уникнути обробки всіх точок одразу
    all_points = np.vstack(points_chunks)
    global_triangulation = Delaunay(all_points)
    return global_triangulation

# Генерація випадкового набору точок
np.random.seed(42)
points = np.random.rand(3000000, 2)

n_jobs = 3
start_time = time.time()  # Початок вимірювання часу

# Розбиваємо точки на менші частини
points_chunks = np.array_split(points, n_jobs)

# Використовуємо Parallel для паралельного обчислення тріангуляцій
triangulations = Parallel(n_jobs=n_jobs)(delayed(compute_delaunay)(chunk) for chunk in points_chunks)

# Об'єднання тріангуляцій
global_triangulation = merge_triangulations(triangulations, points_chunks)

end_time = time.time()  # Кінець вимірювання часу
execution_time = end_time - start_time
print("Час виконання програми: {:.5f} секунд".format(execution_time))

# Візуалізація результату
plt.triplot(points[:, 0], points[:, 1], global_triangulation.simplices)
plt.plot(points[:, 0], points[:, 1], 'o')
plt.title('Триангуляція Делоне')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
