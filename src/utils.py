import random
import cv2
import numpy as np
import os

indice = '1'

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


def save(name, img, id='999'):
    path = buildPath(id, path=name)
    cv2.imwrite(path, img)


def removeSombras(img):
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((3,3), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)

        diff_img2 = np.zeros(diff_img.shape)
        norm_img = cv2.normalize(diff_img, dst=diff_img2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(norm_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    #result_norm = cv2.merge(result_norm_planes)

    return result
    


def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	
    return cv2.resize(image,(width, height), interpolation = cv2.INTER_CUBIC)
    """
    dim = None
	(h, w) = image.shape[:2]

	if width is None and height is None:
		return image

	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)

	else:
		r = width / float(w)
		dim = (width, int(h * r))

	return cv2.resize(image, dim, interpolation = inter)
    """
def show(mat):
    show("window", mat)

def show(label, mat):
    img2 = cv2.resize(mat.copy(), (0, 0), fx = 0.5, fy = 0.5)
    cv2.imshow(label, img2)
    cv2.waitKey(0)        

def color():
    return (random.randint(1,255), random.randint(1,255),random.randint(1,255))

def buildPath(id, path=''):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../static/'+id+'/'+path)
    return filename

def buildPathRoot():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../static/')
    return filename

def removeContornosPqnosImg(img):
    novaImg = np.zeros(img.shape, dtype = "uint8")
    
    #show("window", img)
    #   img = dilatation(img, ratio=0.05)
    #show("window", img)

    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i,c in enumerate(contours):
        tamanhoContorno = cv2.contourArea(c)
        #print('Contorno encontrado tamanho ' + str(tamanhoContorno))
        if tamanhoContorno > 20:
            cv2.drawContours(novaImg, [c], -1, 255, -1)

    #novaImg = cv2.blur(novaImg, (5,5))
    novaImg = dilatation(novaImg, ratio=0.3)
    
    im2, contours, hierarchy = cv2.findContours(novaImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    novaImg = np.zeros(img.shape, dtype = "uint8")
    cv2.drawContours(novaImg, contours, -1, 255, -1)

    return novaImg

def ajustaEspacosContorno(contours, img):
    #print("Contornos encontrados " + str(len(contours)))
    if (len(contours) == 1):
        #print('Retornou contorno 1')
        return contours, img
    else:
        novaMat = np.zeros(img.shape, dtype = "uint8")
        contours = sorted(contours, key=functionSortPrimeiroEsquerdaParaDireita)
        
        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)        
            if (x == 0):
                raise AppException("Erro ao ajustar contornos.")
            if (i == 1):
                c = c - [10,0]
            cv2.drawContours(novaMat, [c], -1, 255, -1)

        #cv2.imshow('img', novaMat)
        #cv2.waitKey(0)

        im2, contours, hierarchy = cv2.findContours(novaMat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return ajustaEspacosContorno(contours, img)

def functionSortPrimeiroEsquerdaParaDireita(c):
    x, y, w, h = cv2.boundingRect(c)
    return x
    