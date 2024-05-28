# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
 

# cube = np.ones((3,3,3),dtype = 'bool')

# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.set_facecolor("Cyan")
# ax.voxels(cube,facecolors="#E02050",edgecolors='k')
# ax.axis('off')
# plt.show()

image_path = 'D:/LNU/6.2/diploma/program/kornia/one_photo_boxed_crop.jpg'

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

def plot_cube(cube_definition, image_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Создание полигональной коллекции граней куба
    polygons = []
    for i in range(len(cube_definition)):
        polygons.append(list(zip(cube_definition[i][0], cube_definition[i][1], cube_definition[i][2])))

    # Отрисовка граней куба
    ax.add_collection3d(Poly3DCollection(polygons, facecolors='#E02050', linewidths=1, edgecolors='k'))

    # Загрузка изображения
    image = plt.imread(image_path)
    imbox = OffsetImage(image, zoom=0.1)  # Уменьшаем масштаб изображения
    ab = AnnotationBbox(imbox, (1, 1, 1), frameon=False)
    ax.add_artist(ab)

    # Настройка осей
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 3])
    ax.set_zlim([0, 3])
    ax.set_facecolor("cyan")
    ax.axis('off')

    plt.show()

# Определение куба
cube_definition = [
    [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],  # Нижняя грань
    [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)],  # Верхняя грань
    [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)],  # Левая грань
    [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],  # Правая грань
    [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],  # Задняя грань
    [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]   # Передняя грань
]

# Путь к изображению
# image_path = 'your_image_path.jpg'  # Укажите путь к вашему изображению

plot_cube(cube_definition, image_path)
