import numpy as np
import cv2 as cv

color = cv.imread('star.png',-1)

img = cv.imread('star.png',0)
ret,thresh = cv.threshold(img,127,255,0)

cv.imwrite('th.jpg', thresh)

im2,contours,hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
#M = cv.moments(cnt)



rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
print(box)

box = np.int0(box)
print(box)
cv.drawContours(color,[box],0,(0,0,255),2)

cv.imwrite('title.jpg', color)