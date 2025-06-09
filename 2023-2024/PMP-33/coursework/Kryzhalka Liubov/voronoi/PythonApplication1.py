import cv2 as cv
import numpy as np

img=cv.imread('photos/3.png')
cv.imshow('3',img)

canny = cv.Canny(img,125,175)
cv.imshow('Canny Edges',canny)

#imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
#cv.imshow('Canny Edges',thresh)
#contours, hierarchiers = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
contours, hierarchiers = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)}')
print(contours)


l=[]
ll=[]
filename = ('output.txt')
file1 = open(filename, 'w')
for i in range(len(contours)):
    i_contour = contours[i]
    for d in range(len(i_contour)):
        XY_Coordinates = i_contour[d]
        l.append(XY_Coordinates[0][0])
        l.append(XY_Coordinates[0][1])
        file1.write(str(XY_Coordinates[0][0]) + ' ' +str(XY_Coordinates[0][1]))
        file1.write('\n')
        ll.append(l)
file1.close()

cv.waitKey(0)