import cv2
import numpy as np
from shapely.geometry import Polygon, Point

def holes():
    image = cv2.imread('photos/10.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    external_contours = []
    internal_contours = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100:
            if hierarchy[0][i][3] == -1:
                external_contours.append(contour)
            else:  
                internal_contours.append(contour)

    external_polygons = [Polygon(contour[:, 0, :]) for contour in external_contours]
    internal_polygons = [Polygon(contour[:, 0, :]) for contour in internal_contours]



    print("External Contours:")
    for contour in external_contours:
        print(contour)

    print("\nInternal Contours:")
    for contour in internal_contours:
        print(contour)

    cv2.drawContours(image, external_contours, -1, (0, 255, 0), 3)
    cv2.drawContours(image, internal_contours, -1, (255, 0, 0), 3)

    cv2.imshow('Detected Non-Simply Connected Region', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return external_contours, internal_contours

