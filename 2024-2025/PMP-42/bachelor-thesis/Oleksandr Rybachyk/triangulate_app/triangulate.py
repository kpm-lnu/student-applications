import numpy as np
import matplotlib.pyplot as plt
import rasterio
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay, KDTree
from joblib import Parallel, delayed
import triangle as tr
import time
import argparse

def process_dem_file(filename, subset_size=0):
    with rasterio.open(filename) as dataset:
        dem_data = dataset.read(1)
        nrows, ncols = dem_data.shape
        x, y = np.meshgrid(np.arange(ncols), np.arange(nrows))

    # Обробка всієї області, якщо subset_size не вказано
    if subset_size==0:
        x_sub = x
        y_sub = y
        z_sub = dem_data
    else:
        # Обробка підмножини
        center_x, center_y = ncols // 2, nrows // 2
        sx, ex = center_x - subset_size // 2, center_x + subset_size // 2
        sy, ey = center_y - subset_size // 2, center_y + subset_size // 2
        x_sub = x[sy:ey, sx:ex]
        y_sub = y[sy:ey, sx:ex]
        z_sub = dem_data[sy:ey, sx:ex]

    points = np.column_stack((x_sub.ravel(), y_sub.ravel(), z_sub.ravel()))

    points = np.column_stack((x_sub.ravel(), y_sub.ravel(), z_sub.ravel()))
    # Функція для виконання тріангуляції Делоне
    def compute_delaunay(points_subset):
        return tr.triangulate({'vertices': points_subset[:, :2]})
    if(subset_size >1600 or subset_size == 0):
        # Поділяємо набір точок на кілька частин для розпаралелювання
        num_parts = 4  # Кількість частин для розпаралелювання
        split_points = np.array_split(points, num_parts)

        # Початок вимірювання часу
        start_time = time.time()

        # Виконуємо паралельне обчислення тріангуляції для кожної частини
        results = Parallel(n_jobs=-1)(delayed(compute_delaunay)(part) for part in split_points)

        # Кінець вимірювання часу
        end_time = time.time()
        execution_time = end_time - start_time
        print("Час виконання програми: {:.5f} секунд".format(execution_time))

        def find_boundary_edges(triangulation):
            """
            Знаходить граничні ребра тріангуляції.
            """
            edges = set()
            for simplex in triangulation['triangles']:  # Використовуємо 'triangles', а не 'simplices'
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
                for simplex in tri['triangles']:  # Використовуємо 'triangles', а не 'simplices'
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
                    new_triangle = sorted([p1, p2, p3])  # Тут може бути проблема, якщо p3 не скаляр
                    distances = [np.linalg.norm(unique_points[new_triangle[i]] - unique_points[new_triangle[j]]) for i in range(3) for j in range(i+1, 3)]
                    
                    if np.all(np.array(distances) < 0.1):  # Використовуємо np.all() замість all()
                        new_triangles.append(new_triangle)

            all_simplices.extend(new_triangles)
            
            # Повертаємо нову тріангуляцію
            return tr.triangulate({'vertices': unique_points, 'triangles': all_simplices}), unique_points

        # Об'єднуємо тріангуляції
        combined_tri, all_points = combine_triangulations(results)
    else:
        start_time = time.time()  # Початок вимірювання часу

        # Тріангуляція за допомогою бібліотеки triangle
        combined_tri = compute_delaunay(points)

        end_time = time.time()  # Кінець вимірювання часу
        execution_time = end_time - start_time
        print("Час виконання програми: {:.5f} секунд".format(execution_time))

    return points, combined_tri['triangles']

if __name__ == "__main__":
    # Парсимо аргументи командного рядка
    parser = argparse.ArgumentParser(description="Обробка DEM файлу та тріангуляція Делоне.")
    parser.add_argument("filename", type=str, help="Шлях до файлу DEM")
    parser.add_argument("--subset_size", type=int, default=300, help="Розмір підмножини (по замовчуванню 300)")
    
    args = parser.parse_args()
    
    # Викликаємо основну функцію з переданими аргументами
    process_dem_file(args.filename, args.subset_size)
