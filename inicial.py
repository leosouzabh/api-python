import cv2
from src import extrator
from src import processamento

def main():
    lista = extrator.extrai('bloco3.jpg')
    itmA = lista[0]
    for idx,item in enumerate(lista):
        #cv2.imshow('sign_{}.jpg'.format(idx), item['img'])
        distancia = processamento.comparaTuple(itmA, item)
        print(distancia)

    cv2.waitKey(0) 
    
if __name__ == '__main__':
	main()
