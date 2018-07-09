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
    utils.save('pb1.jpg', imgGray)
    #imgGray = rotate_bound(imgGray, 90)
    #utils.save('pb2.jpg', imgGray)

    #imgGray = cv2.blur(imgGray, (blurI, blurI))
    #utils.save('blur.jpg', imgGray)

    imgGray, contours =  extraiContornos(imgGray)
    utils.save('thr.jpg', imgGray)
    cnts = sorted(contours, key=functionSort, reverse=True)[0:5]
    
    printaContornoEncontrado(imgOriginal, cnts)

    originalEmGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    #originalHisto = cv2.equalizeHist(originalEmGray)
    originalHisto = originalEmGray
    lista = dict()
    cntArr = dict()

    for i, c in enumerate(cnts):
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            b = 10
            roi = imgOriginal[y-b:y + h+b, x-b:x + w+b]
            utils.save('roi_{}.jpg'.format(i), roi)
            #utils.save('_1_hist_{}.jpg'.format(i), roi)

            #resized = utils.resize(roi, w, interpolation = cv2.INTER_AREA)
            resized = roi
            
            #resized = cv2.blur(resized, (blurI,blurI))
            #utils.save('__{}_blur1.jpg'.format(i), resized)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            #resized = cv2.blur(resized, (5,5))
            retval, resized = cv2.threshold(resized, 120, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            utils.save('t_{}.jpg'.format(i), resized)
            resized = utils.dilatation(resized, ratio=1.0)
            
            utils.save('t1_{}.jpg'.format(i), resized)
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
            
            #lista[i] = mahotas.features.zernike_moments(novaMat, 21)
            lista[i] = cnts
            cntArr[i] = cnts
            utils.save('_img_{}.jpg'.format(i), novaMat)
            

    #utils.show(color)

    hd = cv2.createHausdorffDistanceExtractor()
    sd = cv2.createShapeContextDistanceExtractor()

    
    for idx1 in range(0,5):
        item1 = lista[idx1]
        soma = 0
        for idx2 in range(0,5):
            item2 = lista[idx2]
            #match = dist.euclidean(item1, item2)
            #match = hd.computeDistance(item1, item2)
            match = sd.computeDistance(item1, item2)
            soma += match
            #match = cv2.matchShapes(cntArr[idx1], cntArr[idx2], 1, 0.0)
            #match = round(match, 4)
            print('{} vs {}   ==   {}'.format(idx1, idx2, match) )
        print('Soma: ' + str(soma))
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

    utils.save('contornado.jpg', imgContorno)

#TODO ajustar esse metodo
def ajustaContorno(contours2):
    print('Qtde('+str(i)+')'  + str(len(contours2)))
    if ( i == 2 ):
        for i2, c2 in enumerate(contours2):
            if (cv2.contourArea(c2) > 790):
                debugMat = np.zeros(roi.shape, dtype = "uint8")
                cv2.drawContours(debugMat, [c2], -1, 255, -1 )
                cv2.imshow("t1", debugMat)

                c2 -= (0,10)
                debugMat = np.zeros(roi.shape, dtype = "uint8")
                cv2.drawContours(debugMat, [c2], -1, 255, -1 )

                cv2.imshow("t2", debugMat)
                cv2.waitKey(0)
        print()
    
    cnts = sorted(contours2, key=functionSort, reverse=True)[0]


def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def rotateAndScale(img, scaleFactor = 0.5, degreesCCW = 30):
    (oldY,oldX) = img.shape #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.

    #choose a new image size.
    newX,newY = oldX*scaleFactor,oldY*scaleFactor
    #include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

    #the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    #So I will find the translation that moves the result to the center of that region.
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
    return rotatedImg

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

    for x in range(1,6):
        utils.indice = str(x)
        print('Arquivo ' + utils.indice)
        extrai('../bloco'+str(utils.indice)+'.jpg')
        print('====================================')
        print()