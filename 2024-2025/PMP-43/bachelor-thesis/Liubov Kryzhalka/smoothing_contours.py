import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from functions import read_contours_from_file, write_contours_to_file

def smooth_contour_gaussian_periodic(contour, sigma=2):
    if contour.ndim == 3:
        contour = contour[:, 0]

    if not np.all(contour[0] == contour[-1]):
        contour = np.vstack([contour, contour[0]])

    x = contour[:, 0]
    y = contour[:, 1]

    x_looped = np.r_[x[-sigma:], x, x[:sigma]]
    y_looped = np.r_[y[-sigma:], y, y[:sigma]]

    x_smooth = gaussian_filter1d(x_looped, sigma, mode='wrap')[sigma:-sigma]
    y_smooth = gaussian_filter1d(y_looped, sigma, mode='wrap')[sigma:-sigma]

    x_smooth[-1] = x_smooth[0]
    y_smooth[-1] = y_smooth[0]

    smoothed = np.array([[[int(round(px)), int(round(py))]] for px, py in zip(x_smooth, y_smooth)], dtype=np.int32)
    return smoothed


contours = read_contours_from_file("./data/smoothed_coordinates_of_one_tumor.txt")
print(f"Зчитано {len(contours)} контурів")

smoothed_contours = []
for contour in contours:
    smoothed = smooth_contour_gaussian_periodic(contour, sigma=2)
    smoothed_contours.append(smoothed)

contours = smoothed_contours 

write_contours_to_file(contours, "./data/smoothed_coordinates_of_one_tumor.txt")

plt.figure(figsize=(6, 6))
for contour in contours:
    xs = contour[:, 0] 
    ys = contour[:, 1]
    plt.plot(xs, ys, linewidth=1)

plt.title("Contours from TXT")
plt.gca().invert_yaxis()
plt.axis("equal")
plt.grid(True)
plt.show()