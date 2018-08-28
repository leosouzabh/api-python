import cv2 
import src.utils as utils
import numpy as np



def extraiContornos(imgGray, identificador):
    utils.save('cnh_antesTh.jpg', imgGray, id=identificador)
    retval, imgGray = cv2.threshold(imgGray, 2, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    utils.save('cnh_postTh.jpg', imgGray, id=identificador)
    
    imgGray = utils.removeContornosPqnosImg(imgGray)
    utils.save('cnh_novosContornos.jpg', imgGray, id=identificador)

    imgGray = utils.dilatation(imgGray, ratio=0.2)
    im2, contours, hierarchy = cv2.findContours(imgGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return imgGray, contours, hierarchy

def validaAssinaturaCnh(cnhColor, square1, identificador):
    cnhColor = utils.removeSombras(cnhColor)
    utils.save('cnh_semSombra.jpg', cnhColor, id=identificador)

    imgGray = cnhColor #cv2.cvtColor(cnhColor, cv2.IMREAD_GRAYSCALE)
    #imgGray = cv2.medianBlur(imgGray, 21)

    imgTh, contours, hierarchy =  extraiContornos(imgGray, identificador)
    contours, resized = utils.ajustaEspacosContorno(contours, imgTh)
    utils.save('cnh_resized.jpg', resized, id=identificador)

    
    cnts = contours[0]
    
    novaMat = np.zeros(imgGray.shape, dtype = "uint8")
    cv2.drawContours(novaMat, [cnts], -1, 255, -1)

    xA, yA, wA, hA = cv2.boundingRect(cnts)
    square = novaMat[yA  :yA + hA, xA : xA + wA ]
    utils.save('cnh_square.jpg', square, id=identificador)



    h, w = square1.shape
    
    resized = utils.resize(square, width=w, height=h)
    utils.save('_img_6.jpg', resized, id=identificador)



    path = utils.buildPath(identificador, path="_img_6.jpg")
    print("Novo path " + path)
    imgGray = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    print(imgGray)
    #imgGray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)    
    
    retval, imgGray = cv2.threshold(imgGray, 2, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    imgGray = utils.removeContornosPqnosImg(imgGray)
    im2, contours, hierarchy = cv2.findContours(imgGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours, resized = utils.ajustaEspacosContorno(contours, imgTh)
    cnts2 = contours[0]




    print("Total de contornos CNH ANTES:  " + str(len(cnts)))
    print("Total de contornos CNH DEPOIS: " + str(len(cnts2)))
    

    return cnts2, resized