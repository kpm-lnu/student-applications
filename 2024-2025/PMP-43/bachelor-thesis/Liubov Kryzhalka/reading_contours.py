import numpy as np 
import cv2 
from functions import write_contours_to_file

img = cv2.imread('./photos/2.png') 
  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
ret, thresh = cv2.threshold(gray, 0, 255, 
                            cv2.THRESH_BINARY_INV +
                            cv2.THRESH_OTSU) 
cv2.imshow('image', thresh) 

kernel = np.ones((3, 3), np.uint8) 
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
                            kernel, iterations = 2) 

bg = cv2.dilate(closing, kernel, iterations = 1) 
  
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0) 
ret, fg = cv2.threshold(dist_transform, 0.02
                        * dist_transform.max(), 255, 0) 

fg_uint8 = np.uint8(fg)

contours, _ = cv2.findContours(fg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

cv2.imshow("Contours", contour_img)

write_contours_to_file(contours, "./data/contours_coordinates_ex1.txt", 120)

cv2.waitKey(0)
cv2.destroyAllWindows()
