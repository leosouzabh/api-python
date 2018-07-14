import os
import cv2 
import numpy as np
import mahotas
from scipy.spatial import distance as dist
import src.utils as utils
import src.names as names

blurI = 5

def putText(img, text, point):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = point
    fontScale              = 1
    fontColor              = 255
    lineType               = 2

    cv2.putText(img,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
def extrai(path, identificador):
    color = cv2.imread(path, -1)
    color = cv2.resize(color, (0, 0), fx = 0.3, fy = 0.3)
    imgOriginal = color.copy()
    color = utils.removeSombras(color)
    utils.save('semSombra.jpg', color, id=identificador)

    imgOriginal, color = recuperaAreaAssinada(color.copy(), imgOriginal, identificador)
    
    utils.save('antesGray.jpg', color, id=identificador)
    imgGray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    utils.save('pb1.jpg', imgGray, id=identificador)
    #imgGray = rotate_bound(imgGray, 90)
    #utils.save('pb2.jpg', imgGray)

    #imgGray = cv2.blur(imgGray, (blurI, blurI))
    #utils.save('blur.jpg', imgGray)
    
    utils.save('AntesThr.jpg', imgGray, id=identificador)
    imgGray, contours, hierarchy =  extraiContornos(imgGray, identificador)
    utils.save('thr.jpg', imgGray, id=identificador)
    cnts2 = sorted(contours, key=functionSort, reverse=True)[0:5]
    
        

    printaContornoEncontrado(imgOriginal, cnts2, identificador)
    cnts2 = sorted(cnts2, key=functionSortPrimeiroPapel)
    printaOrdem(imgOriginal, cnts2, identificador)

    originalEmGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    #originalHisto = cv2.equalizeHist(originalEmGray)
    originalHisto = originalEmGray
    lista = dict()
    cntArr = dict()

    for i, c in enumerate(cnts2):
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            b = 10
            #print('{} x={} - y{}'.format(i,x,y))
            roi = imgOriginal[y-b:y + h+b, x-b:x + w+b]
            utils.save('roi_{}.jpg'.format(i), roi, id=identificador)
            #utils.save('_1_hist_{}.jpg'.format(i), roi)

            #roi = utils.resize(roi, width=300, height=300)
            resized = roi.copy()
            
            #resized = cv2.blur(resized, (blurI,blurI))
            #utils.save('__{}_blur1.jpg'.format(i), resized)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            #resized = cv2.blur(resized, (5,5))
            retval, resized = cv2.threshold(resized, 120, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            utils.save('t_{}.jpg'.format(i), resized, id=identificador)
            resized = utils.dilatation(resized)
            
            utils.save('t1_{}.jpg'.format(i), resized, id=identificador)
            im2, contours2, hierarchy = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cnts = sorted(contours2, key=functionSort, reverse=True)[0]
 
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            novaMat = np.zeros(roi.shape, dtype = "uint8")
            cv2.drawContours(novaMat, [cnts], -1, 255, -1)
            #novaMat = cv2.resize(novaMat, (200,200), interpolation = cv2.INTER_AREA)
            
            #lista[i] = mahotas.features.zernike_moments(novaMat, 21)
            lista[i] = cnts
            cntArr[i] = cnts
            utils.save('_img_{}.jpg'.format(i), novaMat, id=identificador)
            

    #utils.show(color)

    hd = cv2.createHausdorffDistanceExtractor()
    sd = cv2.createShapeContextDistanceExtractor()

    out = ""
    resultadoApi = True
    imgResultado = imgOriginal.copy()
    for idx1 in range(0,1):
        item1 = lista[idx1]
        soma = 0
        for idx2 in range(0,5):
            item2 = lista[idx2]
            #match = dist.euclidean(item1, item2)
            #match = hd.computeDistance(item1, item2)
            #match = cv2.matchShapes(cntArr[idx1], cntArr[idx2], 1, 0.0)
            
            ida = sd.computeDistance(item1, item2)
            volta = sd.computeDistance(item2, item1)
            #soma += match
            
            ida = round(ida, 5)
            volta = round(volta, 5)
            out += '{} vs {}   ==   {}-{}\n'.format(idx1, idx2, ida, volta) 
            
            
            #BGR
            if ( idx2 == 0 ):
                imgResultado = contorna(imgResultado, cnts2[idx2], (255,0,0))

            elif ( ida < 10 and volta < 10 ):
                imgResultado = contorna(imgResultado, cnts2[idx2], (0,255,0))
            #elif (soma >= 9 and soma < 20):
            #    imgResultado = contorna(imgResultado, cnts2[idx1], (0,165,255))
            else:
                imgResultado = contorna(imgResultado, cnts2[idx2], (0,0,255))
                resultadoApi = False

        pathTxt = utils.buildPath(identificador, path="calc.txt")
        with open(pathTxt, "w") as text_file:
            text_file.write(out)

    utils.save(names.RESULTADO, imgResultado, id=identificador)
    
    return resultadoApi


def percent(indice):
    tolerancia = 0.01
    if (indice > tolerancia):
        return round((indice-tolerancia) * 100 / tolerancia)
    else:
        return 0


def extraiContornos(imgGray, identificador):
    retval, imgGray = cv2.threshold(imgGray, 0, 220, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    imgGray = utils.dilatation(imgGray)
    im2, contours, hierarchy = cv2.findContours(imgGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return imgGray, contours,hierarchy


def functionSort(c):
    x, y, w, h = cv2.boundingRect(c)
    return w * h

def functionSortPrimeiroPapel(c):
    x, y, w, h = cv2.boundingRect(c)
    return y


def contorna(img, contorno, cor):
    cv2.drawContours(img, [contorno], -1, cor, 4)
    return img

def printaContornoEncontrado(img, cnts, identificador):
    imgContorno = img.copy()

    for idx1,c in enumerate(cnts):
        cor = utils.color()
        #print(cor)
        cv2.drawContours(imgContorno, [c], -1, cor, 4)


def recuperaAreaAssinada(canny_img, imgOriginal, identificador):
    color = canny_img.copy()
    canny_img = cv2.cvtColor(canny_img, cv2.COLOR_BGR2GRAY)
    
    canny_img, contours, hierarchy =  extraiContornos(canny_img, identificador)
    contours = sorted(contours, key=functionSort, reverse=True)[0:5]

    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    height, width = canny_img.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    print('h{} w{}'.format(height, width))

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        if cv2.contourArea(contour) > 400:
            (x,y,w,h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)
        
    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(canny_img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        m = 30
        #print('{} x={} - y{}'.format(i,x,y))
        a = min_y-m if min_y-m > 0 else 0
        b = max_y+m if max_y+m <= height else height
        c = min_x-m if min_x-m > 0 else 0
        d = max_x+m if max_x+m < width else width
        imgGray = color[a:b, c:d]
        imgOriginal = imgOriginal[a:b, c:d]

    utils.save('contornado.jpg', imgGray, id=identificador)
    return imgOriginal, imgGray

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


def printaOrdem(img, contornos, identificador):
    imgSource = img.copy()
    for i, c in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(c)
        putText(imgSource, str(i), (x,y))
    
    utils.save('ordem.jpg', imgSource, id=identificador)

if __name__ == '__main__':

    for x in range(1,6):
        utils.indice = str(x)
        print('Arquivo ' + utils.indice)
        extrai('../bloco'+str(utils.indice)+'.jpg')
        print('====================================')
        print()