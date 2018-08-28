import cv2
import numpy as np


def extraiAssinatura(path, identificador):
    
    print(path)
    img = cv2.imread(path, 0)
    img2 = img.copy()
    template = cv2.imread('../template/cnh.jpg', 0)
    print(template)

    

    
    valA = matchTemplate('cv2.TM_CCOEFF', img, template)
    valB = matchTemplate('cv2.TM_SQDIFF_NORMED', img, template)

    cv2.waitKey(0)                
            

def matchTemplate(method, img, template):
    method = eval(meth)

    w, h = template.shape[::-1]

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    print(top_left)
    cv2.rectangle(img,top_left, bottom_right, (255,0,0), 3)
