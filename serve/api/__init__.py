""" API Blueprint Application """

import os
from flask import Flask
# from flask_restplus import Api
from flask_cors import CORS
from flask import Blueprint, current_app

# from .api import api_bp
# from .client import client_bp

app = Flask(__name__)
CORS(app)
# api = Api(app)

# app.logger.info('>>> {}'.format(Config.FLASK_ENV))

@app.route('/hello')
def hello(): return 'hello'

# api_bp = Blueprint('api_bp', __name__, url_prefix='/api')


# @api_bp.after_request
# def add_header(response):
#     response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
#     return response


from .config import Config

# Import prediction api (choose only one)

# MusicTransformer API
# from .predict import *

# Multitask API
from .predict_multitask import *