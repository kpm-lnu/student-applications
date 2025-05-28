import numpy as np
import matplotlib.pyplot as plt
import time

def in_circle(a, b, c, d):
    """Перевіряє, чи точка d лежить поза колом, що описаний навколо трикутника abc."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    AB = np.array([bx - ax, by - ay])
    AC = np.array([cx - ax, cy - ay])
    AD = np.array([dx - ax, dy - ay])
    BC = np.array([cx - bx, cy - by])
    BD = np.array([dx - bx, dy - by])
    CD = np.array([dx - cx, dy - cy])
    
    # Обчислюємо квадрати відстаней між точками
    AB_squared = np.dot(AB, AB)
    AC_squared = np.dot(AC, AC)
    AD_squared = np.dot(AD, AD)
    BC_squared = np.dot(BC, BC)
    BD_squared = np.dot(BD, BD)
    CD_squared = np.dot(CD, CD)
    
    # Обчислюємо радіуси описаних кол
    circumradius_ABC = AB_squared * AC_squared * (AB_squared + AC_squared - BC_squared) / (8 * (np.linalg.norm(np.cross(AB, AC)) ** 2))
    circumradius_ABD = AB_squared * AD_squared * (AB_squared + AD_squared - BD_squared) / (8 * (np.linalg.norm(np.cross(AB, AD)) ** 2))
    circumradius_ACD = AC_squared * AD_squared * (AC_squared + AD_squared - CD_squared) / (8 * (np.linalg.norm(np.cross(AC, AD)) ** 2))
    # Перевіряємо, чи точка знаходиться поза описаними колами
    return np.all(np.array([circumradius_ABC, circumradius_ABD, circumradius_ACD]) <= 0)


def delaunay(points):
    """Побудова триангуляції Делоне за допомогою інкрементального алгоритму."""
    # Створення початкового трикутника, який охоплює всі точки
    xmin = np.min(points[:, 0]) - 1
    ymin = np.min(points[:, 1]) - 1
    xmax = np.max(points[:, 0]) + 1
    ymax = np.max(points[:, 1]) + 1
    a = [xmin, ymin]
    b = [xmax, ymin]
    c = [xmin, ymax]
    super_triangle = [a, b, c]
    triangles = [super_triangle]
    
    # Проходимо по кожній точці і додаємо її до триангуляції
    for point in points:
        edges = []
        bad_triangles = []
        # Знаходимо трикутники, які містять нову точку
        for triangle in triangles:
            if in_circle(triangle[0], triangle[1], triangle[2], point):
                bad_triangles.append(triangle)
                edges.append([triangle[0], triangle[1]])
                edges.append([triangle[1], triangle[2]])
                edges.append([triangle[2], triangle[0]])
        # Видаляємо погані трикутники
        for triangle in bad_triangles:
            for triangle_index in range(len(triangles)):
                if np.array_equal(triangles[triangle_index], triangle):
                    del triangles[triangle_index]
                    break
        # Видаляємо дублікати ребер
        edges.sort(key=lambda x: (x[0][0], x[0][1], x[1][0], x[1][1]))
        i = 0
        while i < len(edges) - 1:
            if np.array_equal(edges[i], edges[i + 1]):
                del edges[i]
            else:
                i += 1
        # Додаємо нові трикутники
        for edge in edges:
            new_triangle = [edge[0], edge[1], point]
            triangles.append(new_triangle)
    # Видаляємо трикутники, які містять супертрикутник
    result_triangles = []
    for triangle in triangles:
        if not any(np.all(point == vertex) for point in super_triangle for vertex in triangle):
            result_triangles.append(triangle)
    return result_triangles

# Приклад використання:
# Згенеруємо випадковий набір точок для прикладу
np.random.seed(42)
points = np.random.rand(100000, 2)

# Побудова триангуляції Делоне
start_time = time.time()  # Початок вимірювання часу

triangles = delaunay(points)

end_time = time.time()  # Кінець вимірювання часу
execution_time = end_time - start_time
print("Час виконання програми: {:.20f} секунд".format(execution_time))

triangulation = []
for triangle in triangles:
    indices = [np.where((points == v).all(axis=1))[0][0] for v in triangle]
    triangulation.append(indices)

# Виведення триангуляції
plt.triplot(points[:, 0], points[:, 1], triangulation)
plt.plot(points[:, 0], points[:, 1], 'o')
plt.title('Триангуляція Делоне')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
