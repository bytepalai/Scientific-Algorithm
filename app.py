from flask import Blueprint
from flask_restful import Api
from resources.Yolo import Yolo
from resources.Imagenet import Imagenet
from resources.Imagecaption import Imagecaption
from resources.Faceverification import Faceverification

api_bp = Blueprint('api', __name__)
api = Api(api_bp)

# Route
api.add_resource(Yolo, '/detection')
api.add_resource(Imagenet, '/recognition')
api.add_resource(Imagecaption, '/caption')
api.add_resource(Faceverification, '/verification')
