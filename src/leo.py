import cv2 as cv
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import mahotas
from scipy.spatial import distance as dist

def main():
    #compara('../leo/1.png', '../leo/2.png')
    #compara('../leo/ass.png', '../leo/ass2.png')
    compara('../leo/rub_1.png', '../leo/rub_1.png')
    compara('../leo/rub_1.png', '../leo/rub_2.png')
    compara('../leo/rub_1.png', '../leo/rub_3.png')
    compara('../leo/rub_1.png', '../leo/rub1.png')
    
def compara(imgPathA, imgPathB):    
    imgA, impressoA = extraiContorno(imgPathA)
    imgB, impressoB = extraiContorno(imgPathB)

    """colorA  = resizeFix(cv.imread(imgPathA, cv.IMREAD_COLOR), 450)
    cv.drawContours(colorA, impressoA, -1, (0,255,0), 1)
    cv.drawContours(colorA, impressoB, -1, (0,0,255), 1)
    cv.imshow("imgA", colorA)
    cv.waitKey(0)   """

    pontoA = criaPesquisavel(imgA, impressoA)
    pontoB = criaPesquisavel(imgB, impressoB)
    
    d = dist.euclidean(pontoA, pontoB)
    print(d)

def criaPesquisavel(imgA, contornosMatrixA):
    outlineA = np.zeros(imgA.shape, dtype = "uint8")
    cv.drawContours(outlineA, contornosMatrixA, -1, 255, -1)
    return mahotas.features.zernike_moments(outlineA, 21)

def extraiContorno(path):
    color  = cv.imread(path, cv.IMREAD_COLOR)
    img  = cv.imread(path, cv.IMREAD_GRAYSCALE)
    

    size = 450
    img = resizeFix(img,  size)
    color = resizeFix(color, size)

    #cv.imshow("ori", img)
    
    #img = cv2.resize(img, (2000, 2000), interpolation = cv2.INTER_CUBIC)
    #color = cv2.resize(color, (2000, 2000), interpolation = cv2.INTER_CUBIC)

    #cv.imshow("resize", img)
    #cv.waitKey(0)   

    indice = 4
    kernel = np.ones((indice,indice), np.uint8)
    
    #img = cv.blur(img, (5,5))
    #cv.imshow("blur", img)
    (ret, img) = cv.threshold(img, 120, 255, cv.THRESH_BINARY)    
    #cv.imshow("th", img)
    cv.waitKey(0) 
    im2, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    last = len(contours)
    #print(last)

    cnts = sorted(contours, key = cv.contourArea, reverse = True)[1]

    cv.drawContours(color, [cnts], -1, (0,0,255), 1)
    cv.imwrite( "color.png", color )
    #cv.imshow("color", color)
    #cv.waitKey(0)    

    return img, [cnts]

def resizeFix(image, size):
    return cv2.resize(image, (size,size), interpolation = cv2.INTER_AREA)
    

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized

if __name__ == '__main__':
	main()




