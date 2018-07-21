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
    save(imgGray)

def arredaContorno(img):
    


if __name__ == '__main__':
    extrai()