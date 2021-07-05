from flask import Flask
from flask_restful import Api
from flask_cors import CORS

import os

from .semantic_model import SemanticModel
from .api import PredictionEndpoint, WeightsEndpoint, InfoEndpoint

MODELS_DIR = 'models'


def create_application():
    app = Flask(__name__)
    cors = CORS(app, supports_credentials=True)

    for filename in os.listdir(MODELS_DIR):
        if filename.split('.')[-1] != 'jl':
            continue
        model = SemanticModel(os.path.join(MODELS_DIR, filename))
        api = Api(app, prefix=f'/api/v{model.version}', catch_all_404s=True)
        api.add_resource(PredictionEndpoint, "/predict", resource_class_args=(model,))
        api.add_resource(WeightsEndpoint, "/weights", resource_class_args=(model,))
        api.add_resource(InfoEndpoint, "/", resource_class_args=(model,))

    return app
