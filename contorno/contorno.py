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
    imgGray = cv2.imread('t_0.jpg', -1)
    arredaContorno(imgGray)
    save(imgGray)

def arredaContorno(img):
    shape = img.shape
    novaMat = np.zeros(shape, dtype = "uint8")

    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = removeContornosPqnos(contours)
    print(len(contours))

    
def removeContornosPqnos(cnts):
    retorno = []
    totalRemovidos = 0
    for i,c in enumerate(cnts):
        if cv2.contourArea(c) > 100:
            retorno.append(c)
            totalRemovidos+=1

    print('Total removidos: ' + str(totalRemovidos))
    return retorno



if __name__ == '__main__':
    extrai()


