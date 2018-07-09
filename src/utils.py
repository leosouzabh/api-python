import random
import cv2
import numpy as np
import os

indice = '1'

def dilatation(src, ratio=1):
    dilatation_size = int(12 * ratio)
    #dilatation_type = cv2.MORPH_RECT
    #dilatation_type = cv2.MORPH_CROSS
    dilatation_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilatation_type, 
        (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(src, element)
    #show("dilatado", dilatation_dst)
    return dilatation_dst


def save(name, img):
    path = 'C:/dev/git/python/api/data/'+indice
    if not os.path.exists(path):
        os.makedirs(path)

    cv2.imwrite(os.path.join(path , name), img)


def removeSombras(img):
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

def show(mat):
    show("window", mat)

def show(label, mat):
    cv2.imshow(label, resize(mat, width=700))
    cv2.waitKey(0)        

def color():
    return (random.randint(1,255), random.randint(1,255),random.randint(1,255))

def buildPath(id, path=''):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../data/'+id+'/'+path)
    return filename

