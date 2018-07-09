from src.regras import Regras 

class Processamento():
    def processaImagem(self, imagemBase64):
        bo = Regras()
        
        identificador = bo.criaEstrutura()

        bo.escreveImagem(imagemBase64, identificador)
        
        return "imagemBase64"