import os
import cv2 
import numpy as np
import mahotas
from scipy.spatial import distance as dist
from src.AppException import AppException
import src.utils as utils

blurI = 5

def save(img):
    img2 = cv2.resize(img.copy(), (0, 0), fx = 0.5, fy = 0.5)
    cv2.imshow('path', img2)
    cv2.waitKey(0)

def extrai():
    color = cv2.imread('data\leo.jpg', 1)
    color = cv2.resize(color, (0, 0), fx = 0.3, fy = 0.3)
    color = utils.removeSombras(color)
    
    imgGray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    save(imgGray)


    retval, imgGray = cv2.threshold(imgGray, 2, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    imgGray = utils.removeContornosPqnosImg(imgGray)
    imgGray = utils.dilatation(imgGray, ratio=0.09)
    

    im2, contours, hierarchy = cv2.findContours(imgGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = sorted(contours, key=sortAltura, reverse=True)
    assinaturas = list()

    for i,c in enumerate(cnts2):
        
        x, y, w, h = cv2.boundingRect(c)

        existeEntre = existeEntreAlgumaFaixa(assinaturas, y, h)
        if existeEntre == False:
            assinaturas.append((y, y+h))
            
        print(existeEntre)
        
            

    for ass in assinaturas:
        cv2.rectangle(color, (0, ass[0]), (color.shape[0]+w, ass[1]), (255,0,0), 6)

    save(color)


    #imgGray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    #arredaContorno(color, imgGray)

def existeEntreAlgumaFaixa(lista, y,h):
    for i, ass in enumerate(lista):
        #comeca em alguma faixa       
        if ass[0] < y < ass[1]:

            #verifica se precisa expandir o fim
            if y+h > ass[1]: 
                lista[i] = (ass[0], y+h)
            return True

        #termina em alguma faixa 
        elif ass[0] < y+h < ass[1]: 

            #verifica se precisa expandir o inicio
            if y < ass[0]: 
                lista[i] = (y, ass[1])
            return True

    return False

def printaContorno(c, img):
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (0, y), (x+w, y+h), (255,0,0), 6)
    return img

def sortAltura(contorno):
    x, y, w, h = cv2.boundingRect(contorno)
    return h

if __name__ == '__main__':
    extrai()


