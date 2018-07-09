from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from json import dumps
#from src.processamento import Processamento
import os

app = Flask(__name__)
api = Api(app)

class ProcessamentoRest(Resource):
    def post(self):
        print(request.json['image'])

    def get(self):
        #bo = Processamento()
        #result = {"imagem":bo.processaImagem("/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wICh1c2luZyBJSkcgSlBFRyB2ODApLCBxdWFsaXR5ID0gNzUK/9sAQwAIBgYHBgUIBwcHCQkICgwUDQwLCwwZEhMPFB0aHx4dGhwcICQuJyAiLCMcHCg3KSwwMTQ0NB8nOT04MjwuMzQy/9sAQwEJCQkMCwwYDQ0YMiEcITIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy/8AAEQgAMgAyAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/aAAwDAQACEQMRAD8A9Wbxbbfw20p+pAqB/Fyfw2RP1k/+tXJl5+0Mf/fw/wCFNLXH/POIf8DP+FaGHIi3qXxJYyzWllbKskZ2vKW3BT6AY5NZun+KdUiuPOa6eUHqkhyp/Dt+FeV3V9fWbyNCo/eTSSHeMkjOP6Vdi8SahFbW832QN5mRyODzjjFctWtUT909TDYejyNTWp7EnjC6utwiMAKnDAKePzNRS69qMoIMwAPYRj/CvMvDOoy3fiaWNX8sSQBsDkeveu2MMv8Az8yfgF/wrqjNtJnnTpqMmi9/al9/z8P+Qoqh5En/AD8y/wDjv+FFVzMnlRoGk709l+QNTako8p8RynS1lR4Tv37RxjAyap2uuyxWsK3NoBGD8u3uPzroPGNzBJrpg2MrRRL5hP8AGDkgj6c1kyXtrFprRqT5pI27h0964KyafLbqezh5px5720Oi8J2qtq15eImIxGI1z6k5P8q681V0mKCHSbUWwIjaJWBIwTkZyferTV3QjyqzPJqzU5uSG0UUVRmWmYbAD64zUF0s4XZFgEjrQ8wazJPAMiMp9VyR/wCyn860byLEqlegORW8IqFrrUl+8jzPX7KS7vEYCPdCgUt/GQSevqAMY/4FXM3EMouEWWIgnKoUTeM9iR3Fepa1p0cqidQEl6A9j7H2rnrTTiZzKwKFW2/41E6DnWVRbG8K6hRdN7o3rC4nWzgMxBcp8wHTPtVv7QhAODy23AGaqquxFUdBUE7CIljnHXArtlSjM89VJI0T1orI/tpE+VlJZeCfU0VzfVqnY29rHubdwANDtzjny2/9DWtYcwKTzxRRRPf5s1jt9xmalzZvmseE/wCjg99x/nRRXTR+E5q3xE561WvP9QaKK1Ric5N/r5P94/zooorpRJ//2Q==")}
        #return jsonify(result)
        return "ok2"    

class IndexRest(Resource):
    def get(self):
        return "ok"    

api.add_resource(ProcessamentoRest, '/processamento') 
api.add_resource(IndexRest, '/') 


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)