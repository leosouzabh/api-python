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
from src.AppException import AppException

app = Flask(__name__, static_url_path='/static')
api = Api(app)

class ProcessamentoRest(Resource):
    def post(self):
        try:
            bo = Processamento()
            base64Image = request.json['image']
            result = {"imagem": bo.processaImagem(base64Image)}
            return jsonify(result)
        except AppException as error:
            print(error)
            result = {"erro": True, "message": str(error)}
            return jsonify(result)

    def get(self):
        return "ok"

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
        

class MockRest(Resource):
    def get(self):
        try:
            encoded_string = ''
            image = request.args.get('image')
            image = '__inicial.jpg' if image == None else image
            with open("C:\\dev\\git\\python\\api\\data\\"+image, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                
            bo = Processamento()
            base64Image = encoded_string

            resultado, identificador = bo.processaImagem(base64Image)

            result = {"resultado":resultado, "identificador":identificador}
            return jsonify(result)

        except AppException as error:
            result = {"erro": True, "message": str(error)}
            return jsonify(result)


api.add_resource(ProcessamentoRest, '/processamento') 
api.add_resource(IndexRest, '/') 
api.add_resource(MockRest, '/mock') 

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)