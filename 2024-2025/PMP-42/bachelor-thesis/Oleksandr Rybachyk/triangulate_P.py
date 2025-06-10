import numpy as np
import matplotlib.pyplot as plt
import rasterio
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from joblib import Parallel, delayed
import triangle as tr
import time

# Завантаження DEM даних з файлу .tif
filename = "C:\\Users\\User\\Desktop\\Дані\\2017AchykSuuT1_dem_10cm.tif"  # Вкажіть шлях до вашого DEM файлу
#filename = "C:\\Users\\User\\Desktop\\Дані\\the Yingwang Shan_UTM49N.tif"
#filename = "C:\\Users\\User\\Desktop\\Дані\\Panola2021_DEM.tif"

with rasterio.open(filename) as dataset:
    dem_data = dataset.read(1)  # Читаємо висоти
    transform = dataset.transform  # Трансформація для географічних координат

# Створюємо координатну сітку для висот
nrows, ncols = dem_data.shape
x = np.arange(ncols)
y = np.arange(nrows)
x, y = np.meshgrid(x, y)

# Обмеження області: вибираємо центральну частину DEM
subset_size = 30  # Розмір області
center_x, center_y = ncols // 2, nrows // 2
start_x, end_x = max(0, center_x - subset_size // 2), min(ncols, center_x + subset_size // 2)
start_y, end_y = max(0, center_y - subset_size // 2), min(nrows, center_y + subset_size // 2)

x_subset = x[start_y:end_y, start_x:end_x]
y_subset = y[start_y:end_y, start_x:end_x]
dem_data_subset = dem_data[start_y:end_y, start_x:end_x]

# Перетворюємо координати у списки точок
points_subset = np.column_stack((x_subset.ravel(), y_subset.ravel()))
values_subset = dem_data_subset.ravel()

# Функція для виконання тріангуляції Делоне
def compute_delaunay(points_subset):
    return tr.triangulate({'vertices': points_subset})

# Початок вимірювання часу
start_time = time.time()

# Поділяємо набір точок на кілька частин для розпаралелювання
num_parts = 4  # Кількість частин для розпаралелювання
split_points = np.array_split(points_subset, num_parts)

# Виконуємо паралельне обчислення тріангуляції для кожної частини
results = Parallel(n_jobs=num_parts)(delayed(compute_delaunay)(part) for part in split_points)

# Кінець вимірювання часу
end_time = time.time()
execution_time = end_time - start_time
print("Час виконання програми: {:.5f} секунд".format(execution_time))

def find_boundary_edges(triangulation):
    """
    Знаходить граничні ребра тріангуляції.
    """
    edges = set()
    for simplex in triangulation['triangles']:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
            if edge in edges:
                edges.remove(edge)  # Видаляємо внутрішнє ребро
            else:
                edges.add(edge)  # Додаємо нове граничне ребро
    return edges

def combine_triangulations(results):
    """
    Об'єднує кілька часткових тріангуляцій в одну спільну.
    """
    all_points = np.vstack([tri['vertices'] for tri in results])  # Об'єднання всіх точок
    unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)
    
    # Створюємо KDTree для швидкого пошуку відповідних точок
    tree = KDTree(unique_points)
    
    all_simplices = []
    for tri in results:
        simplices = []
        for simplex in tri['triangles']: 
            new_simplex = [tree.query(tri['vertices'][i])[1] for i in simplex]  # Оновлення індексів
            simplices.append(new_simplex)
        all_simplices.extend(simplices)

    # Шукаємо граничні ребра
    boundary_edges = set()
    for tri in results:
        boundary_edges.update(find_boundary_edges(tri))

    # Генеруємо нові трикутники між межами сусідніх частин
    new_triangles = []
    boundary_edges = list(boundary_edges)
    for i, (p1, p2) in enumerate(boundary_edges[:-1]):
        for p3, p4 in boundary_edges[i+1:]:
            new_triangle = sorted([p1, p2, p3]) 
            distances = [np.linalg.norm(unique_points[new_triangle[i]] - unique_points[new_triangle[j]]) for i in range(3) for j in range(i+1, 3)]
            
            if np.all(np.array(distances) < 0.1):
                new_triangles.append(new_triangle)

    all_simplices.extend(new_triangles)
    
    # Повертаємо нову тріангуляцію
    return tr.triangulate({'vertices': unique_points, 'triangles': all_simplices}), unique_points
    #combined_result = {'vertices': unique_points, 'triangles': np.array(all_simplices)}
    #return combined_result, unique_points


# Об'єднуємо тріангуляції
combined_tri, all_points = combine_triangulations(results)

def is_delaunay(triangulation, points, tol=1e-12):
    """
    Перевірка чи задовольняє тріангуляція умові Делоне.
    Повертає True, якщо всі трикутники задовольняють умову.
    """
    triangles = triangulation['triangles']
    tree = KDTree(points)

    for tri in triangles:
        # Вершини трикутника
        A = points[tri[0]]
        B = points[tri[1]]
        C = points[tri[2]]

        # Обчислюємо центр описаного кола
        def circumcenter(A, B, C):
            D = 2 * (A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
            if abs(D) < tol:
                return None, None
            Ux = ((np.dot(A, A)*(B[1]-C[1]) + np.dot(B, B)*(C[1]-A[1]) + np.dot(C, C)*(A[1]-B[1])) / D)
            Uy = ((np.dot(A, A)*(C[0]-B[0]) + np.dot(B, B)*(A[0]-C[0]) + np.dot(C, C)*(B[0]-A[0])) / D)
            return np.array([Ux, Uy]), np.linalg.norm(A - np.array([Ux, Uy]))

        center, radius = circumcenter(A, B, C)
        if center is None:
            continue  # Перевірку неможливо провести для виродженого трикутника

        # Знаходимо всі точки в радіусі (радіус - tol) навколо центра
        indices = tree.query_ball_point(center, radius - tol)
        
        # Якщо якась точка (не зі складових трикутника) потрапила в коло — порушення умови
        for idx in indices:
            if idx not in tri:
                return False

    return True

if is_delaunay(combined_tri, all_points):
    print("Тріангуляція відповідає умові Делоне.")
else:
    print("Тріангуляція не відповідає умові Делоне!")


# Візуалізація об'єднаної тріангуляції
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Відображаємо тріангуляцію у 3D
ax.plot_trisurf(points_subset[:, 0], points_subset[:, 1], dem_data_subset.ravel(),
                triangles=combined_tri['triangles'], cmap='terrain', edgecolor='gray', linewidth=0.5, alpha=0.8)

# Налаштовуємо осі та заголовок
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Elevation (m)')
ax.set_title('3D Delaunay Triangulation of DEM Data (Combined)')
plt.show()

print("Кількість точок до об'єднання:", sum(len(tri['vertices']) for tri in results))
print("Кількість унікальних точок після об'єднання:", len(all_points))
