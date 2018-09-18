import os
import cv2 
import numpy as np
import mahotas
from scipy.spatial import distance as dist
import src.utils as utils
import src.names as names
import src.cnh as cnh
import src.db.database as db
from src.AppException import AppException, QtdeAssinaturasException

blurI = 5
larguraImg = 0
def putText(img, text, point):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = point
    fontScale              = 1
    fontColor              = 255
    lineType               = 2

    cv2.putText(img,text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)


def extrai(path, pathCnh, identificador):

    paramsDb = db.select()
    
    valorAceitavel    = paramsDb[1]
    valorAceitavelCnh = paramsDb[1]
    whTolerancia      = paramsDb[2]
    pxWhiteTolerancia = paramsDb[3]
    
    paramsOut = """ PARAMETROSe
Tolerancia Pontos: {0}
Tolerancia Pontos CNH: {1}
Variacao no tamanho: {2}%
Tolerancia densidade: {3}%\n
    """.format(valorAceitavel, valorAceitavelCnh, whTolerancia, pxWhiteTolerancia)

    densidadeOut = ""

    cnhColor = cv2.imread(pathCnh, -1)
    existeCnh = (cnhColor is not None)
    
    if (existeCnh == True):
        print("Existe")

    color = cv2.imread(path, -1)
    color = cv2.resize(color, (0, 0), fx = 0.3, fy = 0.3)
    imgOriginal = color.copy()
    color = utils.removeSombras(color)
    utils.save('semSombra.jpg', color, id=identificador)

    imgGray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    imgPbOriginal = imgGray.copy()
    utils.save('pb1.jpg', imgGray, id=identificador)
    
    
    imgGray, contours, hierarchy =  extraiContornos(imgGray, identificador)
    utils.save('thr.jpg', imgGray, id=identificador)
    
    
    cnts2 = sorted(contours, key=sortAltura, reverse=True)
    assinaturas = list()

    for i,c in enumerate(cnts2):
        
        x, y, w, h = cv2.boundingRect(c)

        existeEntre = existeEntreAlgumaFaixa(assinaturas, y, h)
        if existeEntre == False:
            assinaturas.append((y-5, y+h+5))
            

    imgCopy = imgOriginal.copy()
    larguraImg = imgOriginal.shape[1]
    for ass in assinaturas:
        cv2.rectangle(imgCopy, (50, ass[0]), (larguraImg-50, ass[1]), (255,0,0), 2)
    utils.save('identificadas_ass.jpg', imgCopy, id=identificador)


    if len(assinaturas) != 5:
        msgEx = "Numero de assinaturas encontradas ({}) é diferente do esperado (5)".format(len(assinaturas))
        raise QtdeAssinaturasException(msgEx, identificador)
    

    assinaturas = sorted(assinaturas)

    lista = dict()

    #ratioDilatacao = recuperaRatioDilatacao(cnts2, imgPbOriginal, identificador)

    for i, ass in enumerate(assinaturas):
        roi = imgPbOriginal[ass[0]:ass[1], 0:larguraImg]
        utils.save('roi_{}.jpg'.format(i), roi, id=identificador)
        
        
        #roi = utils.resize(roi, width=300, height=300)
        resized = roi.copy()
        
        #resized = cv2.blur(resized, (blurI,blurI))
        #utils.save('__{}_blur1.jpg'.format(i), resized)
        
        #resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        #resized = cv2.blur(resized, (5,5))
        retval, resized = cv2.threshold(resized, 120, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        utils.save('th_roi_{}.jpg'.format(i), resized, id=identificador)
        resized, densidade = utils.removeContornosPqnosImg(resized)

        utils.save('t_{}.jpg'.format(i), resized, id=identificador)
        #cv2.waitKey(0) 
        #print('ratioDilatacao ' + str(ratioDilatacao))
        #resized = utils.dilatation(resized, ratio=0.4)
        
        utils.save('t1_{}.jpg'.format(i), resized, id=identificador)
        im2, contours2, hierarchy = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours2, resized = utils.ajustaEspacosContorno(contours2, resized)
        

        cnts = sorted(contours2, key=functionSort, reverse=True)[0]
        

        novaMat = np.zeros(roi.shape, dtype = "uint8")
        cv2.drawContours(novaMat, [cnts], -1, 255, -1)

        xA, yA, wA, hA = cv2.boundingRect(cnts)
        square = novaMat[yA  :yA + hA, xA : xA + wA ]
        utils.save('square_{}.jpg'.format(i), square, id=identificador)
        
        #lista[i] = mahotas.features.zernike_moments(novaMat, 21)
        densidadeOut += "Densidade {} = {}\n".format(i, densidade)
        lista[i] = cnts, ass, square, densidade
        
        

    #utils.show(color)

    hd = cv2.createHausdorffDistanceExtractor()
    sd = cv2.createShapeContextDistanceExtractor()

    out = ""
    outCnh = ""
    sizeOut = ""
    resultadoApi = True
    imgResultado = imgOriginal.copy()

    percentCnh = []

    

    for idx1 in range(0,1): #recupera apenas a primeira imagem e a compara com as outras
        item1   = lista[idx1][0]
        square1 = lista[idx1][2]
        dens1 = lista[idx1][3]
        altura1, largura1 = calculaAlturaLargura(item1)
        soma = 0
        
        item1 = transformaItem(square1, altura1, largura1, identificador, idx1)

        itemCnh = None
        if ( existeCnh == True ):
            itemCnh, squareCnh = cnh.validaAssinaturaCnh(cnhColor, square1, identificador)
            itemCnh = transformaItem(squareCnh, altura1, largura1, identificador, 6)
            print("Contornos img_6 = " + str(len(itemCnh)))

        for idx2 in range(0,5):
            print("Processando imagem " + str(idx2))
            item2 = lista[idx2][0]
            ass   = lista[idx2][1]
            square2 = lista[idx2][2]
            dens2 = lista[idx2][3]
            altura2, largura2 = calculaAlturaLargura(item2)
            sizeOut += 'Dimensao {} x {} \n'.format(largura2, altura2)

            tamanhoCompativel = alturaLarguraCompativel(altura1, largura1, altura2, largura2, whTolerancia)
            densidadeCompativel = calcDensidadeCompativel(dens1, dens2, pxWhiteTolerancia)

            item2 = transformaItem(square2, altura1, largura1, identificador, idx2)

            print("Contornos img_"+str(idx2)+" = " + str(len(item2)))

            #match = hd.computeDistance(item1, item2)
            
            if ( idx1 != idx2 ):
                print("Ida " + str(idx2))
                ida = round(sd.computeDistance(item1, item2), 5)

                print("Volta" + str(idx2))
                volta = round(sd.computeDistance(item2, item1), 5)
            else :
                print("Ida " + str(idx2))
                ida = 0
                print("Volta" + str(idx2))
                volta = 0


            if ( existeCnh == True ):
                print("Ida CNH" + str(idx2))
                idaCnh = round(sd.computeDistance(item2, itemCnh),5)

                print("Volta CNH" + str(idx2))
                voltaCnh = round(sd.computeDistance(itemCnh, item2),5)      
                
                percentSimCnh = calculaSimilaridade(idaCnh, voltaCnh, valorAceitavelCnh)      

                outCnh += '{} ==  {} - {} = {}%\n'.format(idx2, idaCnh, voltaCnh, percentSimCnh) 
                percentCnh.append( percentSimCnh )

            #ida = dist.euclidean(item1, item2)
            #volta = dist.euclidean(item2, item1)
            
            out += '{} vs {} (T{}, D{})  ==   {} - {}\n'.format(idx1, idx2, tamanhoCompativel, densidadeCompativel, ida, volta) 
        
            

            
        
            #BGR
            if ( ida < valorAceitavel and volta < valorAceitavel and tamanhoCompativel == True and densidadeCompativel == True):
                imgResultado = contorna(imgResultado, larguraImg, ass, (0,255,0)) #sucesso
            else:
                imgResultado = contorna(imgResultado, larguraImg, ass, (0,0,255))  #falha
                resultadoApi = False
        
        
        

        pathTxt = utils.buildPath(identificador, path="calc.txt")
        with open(pathTxt, "w") as text_file:
            text_file.write(paramsOut)
            text_file.write('\n')
            text_file.write(sizeOut)
            text_file.write('\n')
            text_file.write(densidadeOut)
            text_file.write('\n')
            text_file.write(out)
            text_file.write('\n')
            text_file.write(outCnh)

    utils.save(names.RESULTADO, imgResultado, id=identificador)
    
    return {'folhaAssinatura':resultadoApi, 'resultadoCnh':False, 'percentCnh':percentCnh}


def calcDensidadeCompativel(dens1, dens2, tolerancia):
    print(dens1)
    print(dens2)
    percent = utils.calculaPercent(int(dens1), int(dens2))
    if (percent <= tolerancia):
        return True
    else:
        return False

def calculaSimilaridade(idaCnh, voltaCnh, tolerancia):
    maior = idaCnh
    if ( voltaCnh > idaCnh ):
        maior = voltaCnh

    return 100 - (maior * 100 / tolerancia)


def alturaLarguraCompativel(altura1, largura1, altura2, largura2, tolerancia):    
    if ( calcPercentual(largura1, largura2) <= tolerancia and calcPercentual(altura1, altura2) <= tolerancia):
        return True
    else:
        return False
    

def calcPercentual(a, b):
    dif  = abs(a - b)
    percent = dif  * 100 /a
    return percent

def calculaAlturaLargura(contorno):
    x, y, w, h = cv2.boundingRect(contorno)
    return h,w

def recuperaRatioDilatacao(contornos, imgOriginal, identificador):
    ratio = 0.1
    for i, c in enumerate(contornos):
        x, y, w, h = cv2.boundingRect(c)
        b = 10
        roi = imgOriginal[y-b:y + h+b, x-b:x + w+b]
        resized = roi.copy()
        #resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        retval, resized = cv2.threshold(resized, 120, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        utils.save('D{}.jpg'.format(i), roi, id=identificador)
        
        preResized = resized.copy()

        for x in range(0, 7):
            resized = utils.dilatation(preResized, ratio=ratio)
        
            utils.save('ratio{}_{}.jpg'.format(i,x), resized, id=identificador)
            im2, contours2, hierarchy = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contours2 = removeContornosPqnos(contours2)



            if (len(contours2) == 1):
                break
            else:
                ratio += 0.3
        
    
    return ratio

def percent(indice):
    tolerancia = 0.01
    if (indice > tolerancia):
        return round((indice-tolerancia) * 100 / tolerancia)
    else:
        return 0


def extraiContornos(imgGray, identificador):
    utils.save('antesTh.jpg', imgGray, id=identificador)
    retval, imgGray = cv2.threshold(imgGray, 2, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    utils.save('postTh.jpg', imgGray, id=identificador)
    
    imgGray, densidade = utils.removeContornosPqnosImg(imgGray)
    utils.save('novosContornos.jpg', imgGray, id=identificador)

    #imgGray = aumentaCanvas(imgGray, identificador)
    
    #imgGray = utils.dilatation(imgGray, ratio=1)
    im2, contours, hierarchy = cv2.findContours(imgGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return imgGray, contours, hierarchy

def functionSort(c):
    x, y, w, h = cv2.boundingRect(c)
    return w * h

def functionSortPrimeiroPapel(c):
    x, y, w, h = cv2.boundingRect(c)
    return y





def contorna(img, larguraImg, ass, cor):
    #cv2.drawContours(img, [contorno], -1, cor, 4)
    cv2.rectangle(img, (20, ass[0]), (larguraImg-20, ass[1]), cor, 3)
    return img

def printaContornoEncontrado(img, cnts, identificador):
    imgContorno = img.copy()

    for idx1,c in enumerate(cnts):
        cv2.drawContours(imgContorno, [c], -1, utils.color(), 4)

    utils.save('contorno.jpg', imgContorno, id=identificador)


def recuperaIdxContornoMaisADireita(contours):
    minX = 999999
    minIdx = 99999
    for idx1, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)        
        if (x < minX):
            minX = x
            minIdx = idx1
    return minIdx


        


def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False


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

    
    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        if cv2.contourArea(contour) > 400:
            (x,y,w,h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x+w, max_x)
            min_y, max_y = min(y, min_y), max(y+h, max_y)
        
    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(canny_img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        m = 30
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

def removeContornosPqnos(cnts):
    retorno = []
    totalRemovidos = 0
    for i,c in enumerate(cnts):
        if cv2.contourArea(c) > 200:
            retorno.append(c)
            totalRemovidos+=1

    return retorno

def aumentaCanvas(img, identificador):
    shape = tuple([500+x for x in img.shape])
    novaMat = np.zeros(shape, dtype = "uint8")

    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i2, c2 in enumerate(contours):
        #c2 = c2 + [300,300]
        cv2.drawContours(novaMat, [c2], -1, 255, -1)

    utils.save('redimensionada.jpg', novaMat, id=identificador)
    return novaMat
    
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


def sortAltura(contorno):
    x, y, w, h = cv2.boundingRect(contorno)
    return h

def transformaItem(square2, altura1, largura1, identificador, idx2):

    square2 = utils.resize(square2, width=largura1, height=altura1)
    im2, contours2, hierarchy = cv2.findContours(square2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours2, key=functionSort, reverse=True)[0]
    novaMat = np.zeros(square2.shape, dtype = "uint8")
    cv2.drawContours(novaMat, [cnts], -1, 255, -1)
    
    
    utils.save('_img_{}.jpg'.format(idx2), novaMat, id=identificador)
    
    #recarrega a imagem do disco, suaviza e contorna
    path = utils.buildPath(identificador, path="_img_"+str(idx2)+".jpg")
    imgGray = cv2.imread(path, cv2.COLOR_BGR2GRAY)    
    #utils.save('debug_{}_antes.jpg'.format(idx2), imgGray, id=identificador)
    imgGray = cv2.medianBlur(imgGray, 5)
    utils.save('__{}_depois.jpg'.format(idx2), imgGray, id=identificador)

    retval, imgGray = cv2.threshold(imgGray, 2, 255, type = cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    im2, contours, hierarchy = cv2.findContours(imgGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = contours[0]

    debugMat = np.zeros(imgGray.shape, dtype = "uint8")
    cv2.drawContours(debugMat, [cnts2], -1, 255, -1 )
    utils.save('debug_{}.jpg'.format(idx2), debugMat, id=identificador)
    
    return cnts2

if __name__ == '__main__':

    for x in range(1,6):
        utils.indice = str(x)
        print('Arquivo ' + utils.indice)
        extrai('../bloco'+str(utils.indice)+'.jpg')
        print('====================================')
        print()