from flask_restful import Resource
from flask import request, jsonify, render_template
import status, requests, json
import base64

from resources.classifier import MarketClassification

model = MarketClassification('models/')

class PredictResource(Resource):
    def post(self):
        if request.headers['Content-Type'] == 'application/image':
            req = request.data
            img = base64.b64decode(req)
            response = model.predict(img)
            return response, status.HTTP_200_OK
        else:
            response = {'error': 'Bad header. Please check and try again.'}
            return response, status.HTTP_400_BAD_REQUEST
