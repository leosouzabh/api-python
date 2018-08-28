import cv2 as cv


def main():
    path = 'picture.jpg'
    img  = cv.imread(path, cv.IMREAD_GRAYSCALE)

    

    x = 756
    y = 188
    w = 92
    h = 52

    # x, y, w, h = cv2.boundingRect(c)
    # b = 10
    # roi = imgOriginal[y-b:y + h+b, x-b:x + w+b]
    
    img = img[y:y+h, x:x+w]
    cv.imwrite('img.jpg', img)

    cv.waitKey(0)
    
if __name__ == '__main__':
	main()




