import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np



img = cv.imread('photo.jpg')
cv.imshow("Obj",img)
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


ret, thresh2=cv.threshold(gray_image,150,255,cv.THRESH_BINARY)
contours,hierarchies=cv.findContours(thresh2,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)



sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)

second_largest_contour = sorted_contours[1]
convex_hull = cv.convexHull(second_largest_contour)


height, width = img.shape[:2]


for point in convex_hull:
        
 point[0][1] = height - point[0][1] - 1

   

with open('contours2.txt', 'w') as f:
     
        for point in convex_hull:
          
            x, y = point[0]
           
            f.write(f"{x} {y}\n")
        first_point = convex_hull[0][0]
        f.write(f"{first_point[0]} {first_point[1]}\n")


with open('contours2.txt', 'r') as f:
    lines = f.readlines()
    contours_from_file = []
    for line in lines:
        x, y = map(int, line.strip().split(' '))
        contours_from_file.append([x, y])


contours_from_file = np.array(contours_from_file)


plt.figure()
plt.plot(contours_from_file[:, 0], contours_from_file[:, 1],color='darkblue')

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

cv.waitKey(0)
