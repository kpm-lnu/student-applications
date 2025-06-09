import matplotlib.pyplot as plt
import cv2
import numpy as np

image = cv2.imread("photos/3.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours, find rotated rectangle, obtain four verticies, and draw 
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

lengths = []

for i in contours:

    x_y,width_height,angle_of_rotation = cv2.minAreaRect(i)
    Height = print(width_height[1])
    rect = cv2.minAreaRect(i)
    box = np.intp(cv2.boxPoints(rect))
    image_contours = np.zeros(image.shape)

    # Draw the contours on the empty image
    cv2.drawContours(image, [box], 0, (36,255,12), 3)
    # OR:
    # cv2.polylines(image, [box], True, (36,255,12), 3)

    plt.imshow(image)
    plt.show()
    
