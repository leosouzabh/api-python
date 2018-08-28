from src.regras import Regras 

class Processamento():
    def processaImagem(self, imagemBase64, cnhBase64, cnhDimensoes):
        bo = Regras()
        
        print('Criando estrutura')
        identificador = bo.criaEstrutura()

        print('Escrevendo Imagem')
        bo.escreveImagem(imagemBase64, cnhBase64, cnhDimensoes, identificador)
                

        return bo.iniciaProcessamento(identificador), identificador
        
        