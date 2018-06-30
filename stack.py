import cv2
image = cv2.imread('bloco.jpg')

#--- Image was too big hence I resized it ---
image = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)

#--- Converting image to grayscale ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#--- Performing inverted binary threshold ---
retval, thresh_gray = cv2.threshold(gray, 0, 255, type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

cv2.imshow('sign_thresh_gray', thresh_gray)

#--- finding contours ---
image, contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)

for i, c in enumerate(contours):
    if cv2.contourArea(c) > 100:
        x, y, w, h = cv2.boundingRect(c)
        roi = image[y  :y + h, x : x + w ]
        cv2.imshow('sign_{}.jpg'.format(i), roi)
        cv2.waitKey()

cv2.destroyAllWindows()