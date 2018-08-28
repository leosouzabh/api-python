import os
import cv2 
import datetime
import src.utils as utils
import src.names as names
import base64
import src.extrator as extrator
import src.extratorAssinaturaCnh as extratorAssinaturaCnh

class Regras():
    def criaEstrutura(self):
        identificador = self.identificador()
        os.makedirs(utils.buildPath(identificador))
        return identificador

    def escreveImagem(self, imagemBase64, cnhBase64, cnhDimencoes, identificador):
        with open(utils.buildPath(identificador, path=names.ORIGINAL), "wb") as fh:
            fh.write(base64.b64decode(imagemBase64)) 

        if ( cnhBase64 != "" ):
            pathCnhFull = utils.buildPath(identificador, path=names.CNH_ORIGINAL)
            with open(pathCnhFull, "wb") as fh:
                fh.write(base64.b64decode(cnhBase64)) 

            #extrai assinatura CNH
            cnhFull  = cv2.imread(pathCnhFull, cv2.IMREAD_GRAYSCALE)
            
            x = int(cnhDimencoes[0])
            y = int(cnhDimencoes[1])
            w = int(cnhDimencoes[2])
            h = int(cnhDimencoes[3])
            
            cnhAss = cnhFull[y:y+h, x:x+w]
            utils.save(names.CNH_ASSINATURA, cnhAss, id=identificador)


    def iniciaProcessamento(self, identificador):
        path = utils.buildPath(identificador, names.ORIGINAL)
        pathCnh = utils.buildPath(identificador, names.CNH_ASSINATURA)
        print('Processando : ' + path) 
        return extrator.extrai(path, pathCnh, identificador)

    def iniciaExtracaoAssinatura(self, identificador):
        path = utils.buildPath(identificador, names.ORIGINAL)
        print('Processando : ' + path) 
        return extratorAssinaturaCnh.extraiAssinatura(path, identificador)

    def identificador(self):
        return datetime.datetime.today().strftime('%Y%m%d_%H%M%S-%f')