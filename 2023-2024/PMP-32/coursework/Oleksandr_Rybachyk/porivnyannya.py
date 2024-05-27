import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import triangle as tr
import time

# Функція для вимірювання часу та виконання триангуляції за допомогою triangle
def triangulate_with_triangle(points):
    start_time = time.perf_counter()  # Початок вимірювання часу
    triangulation = tr.triangulate({'vertices': points})
    end_time = time.perf_counter()  # Кінець вимірювання часу
    execution_time = end_time - start_time
    return triangulation, execution_time

# Функція для вимірювання часу та виконання триангуляції за допомогою Delaunay
def triangulate_with_delaunay(points):
    start_time = time.perf_counter()  # Початок вимірювання часу
    triangulation = Delaunay(points)
    end_time = time.perf_counter()  # Кінець вимірювання часу
    execution_time = end_time - start_time
    return triangulation, execution_time

# Згенеруємо випадковий набір точок для прикладу
np.random.seed(42)
points = np.random.rand(1000, 2)

# Виконуємо триангуляцію обома методами
triangulation_triangle, time_triangle = triangulate_with_triangle(points)
triangulation_delaunay, time_delaunay = triangulate_with_delaunay(points)

# Виводимо час виконання
print("Час виконання триангуляції з використанням triangle: {:.5f} секунд".format(time_triangle))
print("Час виконання триангуляції з використанням Delaunay: {:.5f} секунд".format(time_delaunay))

# Виводимо трикутники для обох методів
plt.figure(figsize=(12, 6))

# Перший графік - триангуляція з використанням triangle
plt.subplot(1, 2, 1)
plt.triplot(points[:, 0], points[:, 1], triangulation_triangle['triangles'], c='r')
plt.plot(points[:, 0], points[:, 1], 'o')
plt.title('Триангуляція Делоне (triangle)\nЧас: {:.5f} секунд'.format(time_triangle))
plt.xlabel('X')
plt.ylabel('Y')

# Другий графік - триангуляція з використанням Delaunay
plt.subplot(1, 2, 2)
plt.triplot(points[:, 0], points[:, 1], triangulation_delaunay.simplices, c='b')
plt.plot(points[:, 0], points[:, 1], 'o')
plt.title('Триангуляція Делоне (Delaunay)\nЧас: {:.5f} секунд'.format(time_delaunay))
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
