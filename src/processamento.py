from src.regras import Regras 

class Processamento():
    def processaImagem(self, imagemBase64):
        bo = Regras()
        
        print('Criando estrutura')
        identificador = bo.criaEstrutura()

        print('Escrevendo Imagem')
        bo.escreveImagem(imagemBase64, identificador)
        
        return bo.iniciaProcessamento(identificador), identificador
        
    def extraiAssinatura(self, imagemBase64):
        bo = Regras()
        
        print('Criando estrutura')
        identificador = bo.criaEstrutura()

        print('Escrevendo Imagem')
        bo.escreveImagem(imagemBase64, identificador)
        
        return bo.iniciaExtracaoAssinatura(identificador), identificador