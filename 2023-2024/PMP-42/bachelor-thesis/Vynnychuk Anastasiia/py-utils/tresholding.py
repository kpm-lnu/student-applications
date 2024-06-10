import cv2 as cv

img = cv.imread('./standard_test_images/tulips.png', cv.IMREAD_GRAYSCALE)

ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

cv.imwrite('./out.jpg', th2)

