import os
import cv2 
import numpy as np
import mahotas
from scipy.spatial import distance as dist


blurI = 5

def save(img):
    cv2.imshow('path', img)
    cv2.waitKey(0)

def extrai():
    color = cv2.imread('roi_4.jpg', 1)
    imgGray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)



    arredaContorno(color, imgGray)

def dilatation(src, ratio=0.4):
    dilatation_size = int(12 * ratio)
    #dilatation_type = cv2.MORPH_RECT
    #dilatation_type = cv2.MORPH_CROSS
    dilatation_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilatation_type, 
        (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(src, element)
    #show("dilatado", dilatation_dst)
    return dilatation_dst



def arredaContorno(color, img):
    shape = color.shape
    retval, img = cv2.threshold(img, 2, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img = dilatation(img, 0.2)
    save(img)
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    print('Antes ' + str(len(contours)))
    contours = removeContornosPqnos(contours)
    print('DEpos ' + str(len(contours)))
    
    novaMat = np.zeros(shape, dtype = "uint8")
    novaMat = cv2.drawContours(color, contours, -1, 255, 3)
    save(novaMat)

    doArreda(contours, img)

    print(len(contours))

def doArreda(contours, img):
    if (len(contours) == 1):
        return contours
    else:
        novaMat = np.zeros(img.shape, dtype = "uint8")

        for i, c in enumerate(contours):
            if (i == 1):
                c = c - [10,0]
            cv2.drawContours(novaMat, [c], -1, 255, -1)

        save(novaMat)

        im2, contours, hierarchy = cv2.findContours(novaMat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        doArreda(contours, img)
        


    
def removeContornosPqnos(cnts):
    retorno = []
    totalRemovidos = 0
    for i,c in enumerate(cnts):
        if cv2.contourArea(c) > 100:
            retorno.append(c)
            totalRemovidos+=1

    print('Total removidos: ' + str(len(cnts) - totalRemovidos))
    return retorno



if __name__ == '__main__':
    extrai()


