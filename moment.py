import cv2
import numpy as np
import mahotas
from scipy.spatial import distance as dist

def getMoment(path, i):
    image = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    #print(image.shape)

    image = cv2.copyMakeBorder(image, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value = 0)
    
    retval, thresh = cv2.threshold(image, 120, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    thresh = cv2.bitwise_not(thresh)
    #thresh[thresh > 0] = 255



#    thresh[thresh > 0] = 255
    cv2.imwrite("contorno/borda"+str(i)+".jpg.jpg", thresh)

    #outline = np.zeros(image.shape, dtype = "uint8")
    
    

    im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(cnts))
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

    x, y, w, h = cv2.boundingRect(cnts)
    outline = np.ones((h,w), dtype = "uint8")
    outline = cv2.copyMakeBorder(outline, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value = 0)
    
    cv2.drawContours(outline, [cnts], -1, 255, -1)
    outline = cv2.bitwise_not(outline) 
    cv2.imwrite("contorno/out"+str(i)+".jpg", outline)

    
    moments = mahotas.features.zernike_moments(outline, 21)
    return moments 

momentA = getMoment("contorno/img_a.jpg",1)
momentB = getMoment("contorno/img_b.jpg",2)
momentC = getMoment("contorno/img_c.jpg",3)
momentD = getMoment("contorno/img_d.jpg",4)
momentE = getMoment("contorno/img_e.jpg",5)

arr = [momentA,momentB,momentC,momentD,momentE]
for idx1 in range(0,5):
    for idx2 in range(0,5):
        valA = arr[idx1]
        valB = arr[idx2]
        valorAB = dist.euclidean(valA, valB)
        print('{} x {} = {}'.format(idx1, idx2, round(valorAB,5)) )





