import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import mahotas
from scipy.spatial import distance as dist
import utils

blurI = 5
def extrai(path):
    color = cv2.imread(path, -1)
    color = cv2.resize(color, (0, 0), fx = 0.3, fy = 0.3)
    imgOriginal = color.copy()
    color = utils.removeSombras(color)
    utils.save('semSombra.jpg', color)
    
    imgGray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    #imgGray = adjust_gamma(imgGray, gamma=0.2)

    utils.save('pb.jpg', imgGray)

    imgGray = cv2.blur(imgGray, (blurI, blurI))
    utils.save('blur.jpg', imgGray)

    imgGray, contours =  extraiContornos(imgGray)
    utils.save('thr.jpg', imgGray)
    cnts = sorted(contours, key=functionSort, reverse=True)[0:5]
    
    printaContornoEncontrado(imgOriginal, cnts)

    originalEmGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    #originalHisto = cv2.equalizeHist(originalEmGray)
    originalHisto = originalEmGray
    lista = dict()

    for i, c in enumerate(cnts):
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            b = 30
            roi = imgOriginal[y-b:y + h+b, x-b:x + w+b]
            utils.save('roi_{}.jpg'.format(i), roi)
            #utils.save('_1_hist_{}.jpg'.format(i), roi)

            
            resized = roi
            #resized = cv2.blur(resized, (blurI,blurI))
            #utils.save('__{}_blur1.jpg'.format(i), resized)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            retval, resized = cv2.threshold(resized, 120, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            resized = utils.dilatation(resized, ratio=1)
            im2, contours2, hierarchy = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cnts = sorted(contours2, key=functionSort, reverse=True)[0]

            
            """
            debugMat = np.zeros(roi.shape, dtype = "uint8")
            cv2.drawContours(debugMat, [cnts], -1, 255, -1 )
            debugMat = cv2.resize(debugMat, (700,700), interpolation = cv2.INTER_AREA)
            #utils.show("teste", debugMat)
            """


            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            novaMat = np.zeros(roi.shape, dtype = "uint8")
            cv2.drawContours(novaMat, [cnts], -1, 255, -1)
            #novaMat = cv2.resize(novaMat, (200,200), interpolation = cv2.INTER_AREA)
            
            lista[i] = mahotas.features.zernike_moments(novaMat, 21)
            utils.save('_img_{}.jpg'.format(i), novaMat)
            

    #utils.show(color)

    for idx1 in range(0,5):
        item1 = lista[idx1]
        for idx2 in range(0,5):
            item2 = lista[idx2]
            
            d = dist.euclidean(item1, item2)
            print('{} vs {}   ==   {}% - {}'.format(idx1, idx2, percent(d), round(d,6)) )
        print()

    #print(len(lista))
    return lista


def percent(indice):
    tolerancia = 0.01
    if (indice > tolerancia):
        return round((indice-tolerancia) * 100 / tolerancia)
    else:
        return 0


def extraiContornos(imgGray):
    retval, imgGray = cv2.threshold(imgGray, 10, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    imgGray = utils.dilatation(imgGray)
    im2, contours, hierarchy = cv2.findContours(imgGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return imgGray, contours



def functionSort(c):
    x, y, w, h = cv2.boundingRect(c)
    return w * h


def printaContornoEncontrado(img, cnts):
    imgContorno = img.copy()

    for idx1,c in enumerate(cnts):
        cv2.drawContours(imgContorno, [c], -1, utils.color(), 4)


    #cv2.drawContours(imgContorno, cnts, -1, utils.color(), -1)
    utils.save('contornado.jpg', imgContorno)


def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

if __name__ == '__main__':

    """
    img1 = cv2.imread('../bloco.jpg')
    img1 = utils.removeSombras(img1)
    a = np.double(img1)
    b = a + 30
    img2 = np.uint8(b)
    utils.show("frame",img1)
    utils.show("frame2",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
    """    
    img = cv2.imread('../bloco.jpg', 0)
    img = utils.removeSombras(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    equ = clahe.apply(img)
    #equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    utils.show("frame",res)
    
    gamma = 0.3      
    img = cv2.imread('../bloco.jpg', 0)
    img = utils.removeSombras(img)
    equ = adjust_gamma(img, gamma=0.3)
    #equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    utils.show("frame",res)
    """

    for x in [2]:
        utils.indice = str(x)
        blurI = x
        print()
        print()
        print('Blur {}'.format(x) )
        extrai('../bloco.jpg')