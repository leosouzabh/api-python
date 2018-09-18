import os
import cv2 
import numpy as np


def calcPixel(path):
    img = cv2.imread(path)
    n_white_pix = np.sum(img == 255)
    return n_white_pix


"""
    cem = 100%
    dez = ?     ->     ?  = 100 * dez / cem
"""
def percent(cem, dez):
    return abs(int(100 - (100 * dez / cem)))

imgZero = calcPixel("pixel_img/th_roi_0.jpg")
print('Number of white pixels:', imgZero, "100")  

for x in range(1, 5):
    n_white_pix =  calcPixel("pixel_img/th_roi_"+str(x)+".jpg")
    print('Number of white pixels:', n_white_pix, percent(imgZero, n_white_pix))  