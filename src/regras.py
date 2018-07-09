import os
import datetime
import src.utils as utils
import src.names as names
import base64

class Regras():
    def criaEstrutura(self):
        identificador = self.identificador()
        os.makedirs(utils.buildPath(identificador))
        return identificador

    def escreveImagem(self, imagemBase64, identificador):
        print(imagemBase64)
        with open(utils.buildPath(identificador, path=names.ORIGINAL), "wb") as fh:
            fh.write(base64.b64decode(imagemBase64)) 

    def identificador(self):
        return datetime.datetime.today().strftime('%Y%m%d_%H%M%S')