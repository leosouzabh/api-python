from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from src.processamento import Processamento

app = Flask(__name__)
api = Api(app)

class ProcessamentoRest(Resource):
    def post(self):
        print(request.json['image'])

    def get(self):
        bo = Processamento()
        result = {"imagem":bo.processaImagem("base64")}
        return jsonify(result)

api.add_resource(ProcessamentoRest, '/processamento') 

if __name__ == '__main__':
     app.run(port='5002')