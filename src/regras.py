import os
import datetime
import src.utils as utils
import src.names as names
import base64
import src.extrator as extrator

class Regras():
    def criaEstrutura(self):
        identificador = self.identificador()
        os.makedirs(utils.buildPath(identificador))
        return identificador

    def escreveImagem(self, imagemBase64, identificador):
        with open(utils.buildPath(identificador, path=names.ORIGINAL), "wb") as fh:
            fh.write(base64.b64decode(imagemBase64)) 


    def iniciaProcessamento(self, identificador):
        path = utils.buildPath(identificador, names.ORIGINAL)
        print('Processando : ' + path) 
        extrator.extrai(path, identificador)


    def identificador(self):
        return datetime.datetime.today().strftime('%Y%m%d_%H%M%S')