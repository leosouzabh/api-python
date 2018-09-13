from flask import Flask, request, jsonify, request
from flask_restful import Resource, Api
from json import dumps
from src.processamento import Processamento
from os import listdir
from datetime import datetime
import os
import cv2
import src.utils as utils
import base64 
from src.AppException import AppException, QtdeAssinaturasException
import src.db.database as db

app = Flask(__name__, static_url_path='/static')
api = Api(app)

class ProcessamentoRest(Resource):
    def post(self):
        try:
            bo = Processamento()
            base64Image = request.json['image']
            cnh64Image = request.json['cnh']
            cnhDimensoes = request.json['cnhDimensoes']

            resultado, identificador = bo.processaImagem(base64Image, cnh64Image, cnhDimensoes)
            result = {"resultado":resultado, "erro":False, "identificador":identificador}
            return jsonify(result)
        
        except AppException as error:
            result = {"erro": True, "message": str(error)}
            return jsonify(result)

        except QtdeAssinaturasException as error:
            result = {"erro": True, "message": str(error), "identificador":error.identificador}
            return jsonify(result)

    def get(self):
        return "ok"




class ParamRest(Resource):
    def get(self):
        param = db.select()
        distanciaPontos = param[1]
        tamanho = param[2]
        densidade = param[3]
        return jsonify({
            "distanciaPontos":distanciaPontos,
            "tamanho":tamanho,
            "densidade":densidade
        })

    def post(self):
        distanciaPontos  = request.json['distanciaPontos']
        tamanho          = request.json['tamanho']
        densidade        = request.json['densidade']

        db.update(distanciaPontos, tamanho, densidade)


class IndexRest(Resource):
    def get(self):
        dirname = utils.buildPathRoot()
        dirs = listdir(dirname)

        result = list()
        dirs = sorted(dirs, reverse=True)
        
        for dir in dirs:
            if (dir != 'back'):
                result.append({
                    'cod':dir,
                    'desc':self.parse(dir)
                })

        return jsonify(result)

    def parse(self, path):
        datetime_object = datetime.strptime(path, '%Y%m%d_%H%M%S-%f')
        return datetime_object.strftime('%d/%m %H:%M')
        

'''
    http://localhost/mock?image=cnh.jpg
'''
class MockRest(Resource):
    def get(self):
        try:
            encoded_string = ''
            encoded_cnh = ''

            image = request.args.get('image')
            image = '__inicial.jpg' if image == None else image
            with open("/app/data/"+image, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

            with open("/app/data/cnh/"+image, "rb") as image_file:
                encoded_cnh = base64.b64encode(image_file.read())                
                
            bo = Processamento()
            base64Image = encoded_string
            
            cnhDimensoes = [634, 1003, 1633, 213]
            resultado, identificador = bo.processaImagem(encoded_string, encoded_cnh, cnhDimensoes)
            result = {"resultado":resultado, "erro":False, "identificador":identificador}
            return jsonify(result)
            
            
            
        except AppException as error:
            result = {"erro": True, "message": str(error)}
            return jsonify(result)

        except QtdeAssinaturasException as error:
            result = {"erro": True, "message": str(error), "identificador":error.identificador}
            return jsonify(result)


api.add_resource(ProcessamentoRest, '/processamento') 
#api.add_resource(CnhValidacaoRest, '/cnh') 
api.add_resource(IndexRest, '/') 
api.add_resource(MockRest, '/mock')
api.add_resource(ParamRest, '/param') 

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)