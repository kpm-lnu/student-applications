import cv2
import numpy as np
import matplotlib.pyplot as plt


## TODO. Add abillity to change down_size and scale
def read_image_file(image_path: str, down_size_factor=6, scale=3):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, thresholded = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.array([]), np.array([])

    largest_contour = max(contours, key=cv2.contourArea)

    if down_size_factor > 1:
        largest_contour = largest_contour[::down_size_factor]

    points = largest_contour.squeeze()

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    points = (points - min_vals) / (max_vals - min_vals) * scale

    N = len(points)
    segments = np.stack((np.arange(N), np.arange(N) + 1), axis=1) % N

    return points, segments
