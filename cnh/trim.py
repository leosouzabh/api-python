import cv2
import numpy as np
from PIL import Image, ImageChops

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

def trim(path):
	im = Image.open(path)
	bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
	diff = ImageChops.difference(im, bg)
	diff = ImageChops.add(diff, diff, 2.0, -100)
	bbox = diff.getbbox()
	if bbox:
		return im.crop(bbox).save('saida.jpg')


inputPath = 'saida.jpg'
img = cv2.imread(inputPath,0)
existeCnh =  img is not None 

if (existeCnh == True):
    print("Existe")

#img = removeSombras(img)
#cv2.imwrite('semSombra.jpg', img)
#trim('semSombra.jpg')