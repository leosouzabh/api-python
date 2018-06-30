import cv2 as cv
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import mahotas
from scipy.spatial import distance as dist

def main():
    path = "bloco2.jpg"
    color = removeSombras(path)
    #color = cv2.resize(color, (0, 0), fx = 0.5, fy = 0.5)
    cv2.imwrite('sem.jpg', color)
    imgGray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    imgGray = cv.blur(imgGray, (20, 20))
    #(ret, imgGray) = cv.threshold(imgGray, 220, 255, cv.THRESH_BINARY)
    
    #imgGray = cv.adaptiveThreshold(imgGray, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 33, 2)

    
    retval, imgGray = cv2.threshold(imgGray, 0, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    #imgGray = cv2.copyMakeBorder(imgGray, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value = 255)
    im2, contours, hierarchy = cv.findContours(imgGray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    cnts = sorted(contours, key=functionSort, reverse=True)[0:10]
    print(len(cnts))

    #area = cv.contourArea(cnt)

    print(len(contours))

    cv.drawContours(color, cnts, -1, (0,0,255), 4)

    for i, c in enumerate(cnts):
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            print(w * h)
            roi = color[y  :y + h, x : x + w ]
            cv2.imshow('sign_{}.jpg'.format(i), roi)
            cv2.waitKey()


    show(color)
    
def functionSort(c):
    x, y, w, h = cv2.boundingRect(c)
    return w * h

def removeSombras(path):
    img = cv2.imread(path, -1)

    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
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

def show(mat):
    cv.imshow("resize", resize(mat, width=800))
    cv.waitKey(0)






if __name__ == '__main__':
	main()