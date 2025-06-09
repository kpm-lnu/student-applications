import numpy as np
import matplotlib.pyplot as plt
import rasterio
from mpl_toolkits.mplot3d import Axes3D
import triangle as tr
import time

# Завантаження DEM даних з файлу .tif
#filename = "C:\\Users\\User\\Desktop\\Дані\\2017AchykSuuT1_dem_10cm.tif" # Вкажіть шлях до вашого DEM файлу
filename = "C:\\Users\\User\\Desktop\\Дані\\the Yingwang Shan_UTM49N.tif" 

with rasterio.open(filename) as dataset:
    dem_data = dataset.read(1)  # Читаємо висоти
    transform = dataset.transform  # Трансформація для географічних координат

# Створюємо координатну сітку для висот5
nrows, ncols = dem_data.shape
x = np.arange(ncols)
y = np.arange(nrows)
x, y = np.meshgrid(x, y)

# Обмеження області: вибираємо центральну частину DEM
subset_size = 1600  # Розмір області 
center_x, center_y = ncols // 2, nrows // 2
x_subset = x[center_y - subset_size // 2:center_y + subset_size // 2, center_x - subset_size // 2:center_x + subset_size // 2]
y_subset = y[center_y - subset_size // 2:center_y + subset_size // 2, center_x - subset_size // 2:center_x + subset_size // 2]
dem_data_subset = dem_data[center_y - subset_size // 2:center_y + subset_size // 2, center_x - subset_size // 2:center_x + subset_size // 2]
# Перетворюємо координати у списки точок
points_subset = np.column_stack((x_subset.ravel(), y_subset.ravel()))
values_subset = dem_data_subset.ravel()  # Висоти точок

start_time = time.time()  # Початок вимірювання часу

# Тріангуляція за допомогою бібліотеки triangle
triangulation_subset = tr.triangulate({'vertices': points_subset})

end_time = time.time()  # Кінець вимірювання часу
execution_time = end_time - start_time
print("Час виконання програми: {:.5f} секунд".format(execution_time))

# Увімкнення інтерактивного режиму для обертання
plt.ion()

# 3D Візуалізація тріангуляції
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Відображаємо тріангуляцію у 3D
ax.plot_trisurf(points_subset[:, 0], points_subset[:, 1], values_subset, 
                triangles=triangulation_subset['triangles'], cmap='terrain', 
                edgecolor='gray', linewidth=0.5, alpha=0.8)  # Зменшено товщину ребер (linewidth=0.5)

# Налаштовуємо осі та заголовок
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Elevation (m)')
ax.set_title('3D Triangulation of DEM Data (Subset) using triangle')

# Додаємо можливість обертати графік
plt.show(block=True)
