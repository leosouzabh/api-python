import cv2
import numpy as np
import mahotas
from scipy.spatial import distance as dist

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


def main():
    
    img1 = cv2.imread('a.jpg',0)

    for x in range(1, 45):
        img = rotate_bound(img1, x)
        ret, img = cv2.threshold(img, 200, 255,0)
        cv2.imwrite('data\saida'+str(x)+'.jpg', img)
        #print(x)

    for x in range(1, 45):
        img = rotate_bound(img1, x*-1)
        ret, img = cv2.threshold(img, 200, 255,0)
        cv2.imwrite('data\saida'+str(x+45)+'.jpg', img)

        



if __name__ == '__main__':
    main()    