from flask_restful import Resource, reqparse
from flask import jsonify


class PredictionEndpoint(Resource):

    def __init__(self, model):
        self.model = model
        super().__init__()

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('text', required=True, nullable=False)
        args = parser.parse_args()
        prediction = self.model.predict(args['text'])
        return jsonify(prediction)


class WeightsEndpoint(Resource):

    def __init__(self, model):
        self.model = model
        super().__init__()

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('text', required=True, nullable=False)
        args = parser.parse_args()
        prediction = self.model.weights(args['text'])
        return jsonify(prediction)


class InfoEndpoint(Resource):

    def __init__(self, model):
        self.model = model
        super().__init__()

    def get(self):
        return jsonify(self.model.info)