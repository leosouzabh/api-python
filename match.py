import cv2
import numpy as np
import mahotas
from scipy.spatial import distance as dist

"""

img1 = cv2.imread('a.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('aa.jpg',cv2.IMREAD_GRAYSCALE)

ret, thresh = cv2.threshold(img1, 127, 255,0)
ret, thresh2 = cv2.threshold(img2, 127, 255,0)
item1 = mahotas.features.zernike_moments(img1, 21)
item2 = mahotas.features.zernike_moments(img2, 21)

im2,contours,hierarchy = cv2.findContours(thresh,2,1)
cnt1 = contours[0]

im2,contours,hierarchy = cv2.findContours(thresh2,2,1)
cnt2 = contours[0]


ret1 = dist.euclidean(item1, item2)

ret2 = cv2.matchShapes(cnt1,cnt2,1,0.0)
print (ret1)
print (ret2)

"""



img = cv2.imread('aa.jpg',cv2.IMREAD_GRAYSCALE)
ret, img = cv2.threshold(img, 200, 255,0)
im2,contours,hierarchy = cv2.findContours(img,2,1)
cnt1 = contours[0]

debugMat = np.zeros(img.shape, dtype = "uint8")
print(len(contours))
cv2.drawContours(debugMat, contours, -1, 255, -1 )
cv2.imshow('img', debugMat)
cv2.waitKey(0)

menor = 99
label = 1

maior = 0
labelm = 0

hd = cv2.createHausdorffDistanceExtractor()
sd = cv2.createShapeContextDistanceExtractor()




for x in range(1,89):
    
    img1 = cv2.imread('data\saida'+str(x)+'.jpg', cv2.IMREAD_GRAYSCALE)
    ret, img1 = cv2.threshold(img1, 200, 255,0)
    im2,contours2,hierarchy = cv2.findContours(img1,2,1)
    cnt2 = contours2[0]
    #match = cv2.matchShapes(cnt1,cnt2,1,0.0)

    #match = hd.computeDistance(cnt1,cnt2)
    match = sd.computeDistance(cnt1,cnt2)

    """
    debugMat = np.zeros(img1.shape, dtype = "uint8")
    cv2.drawContours(debugMat, contours2, -1, 255, -1 )
    cv2.imshow('img', debugMat)
    cv2.waitKey(0)
    """

    print(str(x) + ': ' + str(match))

    if ( match <= menor):
        menor = match
        label = x
    
    if (match > maior):
        maior = match
        labelm = x
        

print(menor)
print(label)

print(maior)
print(labelm)







