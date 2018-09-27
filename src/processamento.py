from src.regras import Regras 
import base64
import os

class Processamento():
    def reprocessaImagem(self, id):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../static/'+id+'/__inicial.jpg')
        with open(filename, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            print(encoded_string)
            self.processaImagem(encoded_string, "", "")

        print(filename)

    def processaImagem(self, imagemBase64, cnhBase64, cnhDimensoes):
        bo = Regras()
        
        print('Criando estrutura')
        identificador = bo.criaEstrutura()

        print('Escrevendo Imagem')
        bo.escreveImagem(imagemBase64, cnhBase64, cnhDimensoes, identificador)

        return bo.iniciaProcessamento(identificador), identificador
        
        