from skimage.morphology import medial_axis
from scipy import ndimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import cv2
import numpy as np
from Input import read_from_file


# Зчитуємо зображення за вказаним шляхом
image = cv2.imread('photos/8.png')

# Перетворюємо зображення у чорно-біле
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)


# Інвертуємо маску
mask = 255 - mask

mask = ndimage.binary_fill_holes(mask).astype(int)

skel, distance = medial_axis(mask, return_distance=True)
dist_on_skel = distance * skel

#contour_points = np.argwhere(mask == 0)

lines = read_from_file('output.txt')
contour_points=[]
for i in range(len(lines)):
    contour_points.append([float(lines[i].split()[0]), float(lines[i].split()[1])])

# Отримати відстані від контуру до найближчої точки медіальної осі
#distances_to_skel = distance[contour_points[:, 0], contour_points[:, 1]]

# Отримуємо координати точок медіальної осі
skeleton_points = np.argwhere(skel == 1)

distances_to_skel_contour = []
min_index_sceleton=0
indexes_of_sceleton=[]
# Для кожної точки контуру області
for contour_point in contour_points:
    # Отримайте координати поточної точки контуру
    x_contour, y_contour = contour_point
    
    # Шукайте відстань від поточної точки контуру до найближчої точки медіальної осі
    distances = cdist([(x_contour, y_contour)], skeleton_points)
    
    # Знайдіть мінімальну відстань
    min_distance = np.min(distances)
    indexes_of_sceleton.append(np.argmin(distances))
    # Додайте мінімальну відстань до списку відстаней
    distances_to_skel_contour.append(min_distance)


min_index = np.argmin(distances_to_skel_contour)
index=indexes_of_sceleton[min_index]
# Отримати координати точки медіальної осі, що відповідає найкоротшій відстані
x_skel, y_skel = skeleton_points[index]

# Отримати координати точки контуру, що відповідає найкоротшій відстані
x_contour, y_contour = contour_points[min_index]

print("Координати точки медіальної осі:", x_skel, y_skel)
print("Координати точки на контурі:", x_contour, y_contour )



# Отримуємо координати точок маски
mask_points = np.argwhere(mask == 1)

# Перетворюємо координати в список кортежів
skeleton_points_list = [tuple(point) for point in skeleton_points]
mask_points_list = [tuple(point) for point in mask_points]

x = [point[0] for point in skeleton_points ]
y = [point[1] for point in skeleton_points ]


fig, ax = plt.subplots()#figsize=(8, 8), dpi=200)

# Відображаємо зображення медіальної осі та контури
#ax.imshow(dist_on_skel, cmap='gray', interpolation='nearest')
#ax.contour(mask, [0.5], colors='w')
#ax.set_title('medial_axis')

contour_points=np.array(contour_points)
ax.plot(contour_points[:, 0], contour_points[:, 1], color='b')
ax.plot(x, y, color='b')
#ax.plot([x_skel, x_contour], [y_skel, y_contour], color='red')x_contour, y_contour
#ax.scatter(x_contour, y_contour, color='red',s=5 )
#ax.contour(mask, [0.5], colors='w')
#ax.set_title('medial_axis')

plt.gca().invert_yaxis()

plt.show()