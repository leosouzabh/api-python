import cv2
import numpy as np


def main():
    img = cv2.imread('a.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #th = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #ret,thresh = cv2.threshold(gray,127,255,0)

    val, th = cv2.threshold(gray, 0, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    show(th)



def show(mat, label = "Img"):
    img2 = cv2.resize(mat.copy(), (0, 0), fx = 0.5, fy = 0.5)
    cv2.imshow(label, img2)
    cv2.waitKey(0)        



if __name__ == '__main__':
    main()