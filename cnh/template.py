import cv2
import numpy as np
from matplotlib import pyplot as plt
import os



def show(mat, label = "Img"):
    mat = cv2.resize(mat.copy(), (0, 0), fx = 0.2, fy = 0.2)
    #cv2.imshow(label, img2)
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'saida/')
    cv2.imwrite(filename+label+'.jpg', mat)
    #cv2.waitKey(0)        



files = ['a','b','c']
for fl in files:

    img = cv2.imread(fl+'.jpg',0)
    img2 = img.copy()
    template = cv2.imread('template.jpg',0)
    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list
    #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_SQDIFF_NORMED']
    #methods = ['cv2.TM_CCORR_NORMED']

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(meth, min_val, max_val, min_loc, max_loc)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        ab = (top_left[0], top_left[1] - 250)
        cd = (bottom_right[0], top_left[1])


        novaImg = img[ab[1]:cd[1], ab[0]:cd[0]]
        #show(novaImg, label=fl + ' - ' + meth)

        #cv2.rectangle(img,ab, cd, (255,0,0), 3)

        cv2.rectangle(img,top_left, bottom_right, (0, 255, 0), 3)

        show(img, label=fl + ' - ' + meth)
    
cv2.waitKey(0)                
        