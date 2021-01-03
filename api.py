import sys
import os
import logging

import flask
from flasgger import Swagger
from flask import Flask, request, jsonify, Response
import sys

import pickle
import numpy as np
import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# NOTE this import needs to happen after the logger is configured


# Initialize the Flask application
application = Flask(__name__)

application.config['ALLOWED_EXTENSIONS'] = set(['pdf'])
application.config['CONTENT_TYPES'] = {"pdf": "application/pdf"}
application.config["Access-Control-Allow-Origin"] = "*"


CORS(application)

swagger = Swagger(application)

def clienterror(error):
    resp = jsonify(error)
    resp.status_code = 400
    return resp


def notfound(error):
    resp = jsonify(error)
    resp.status_code = 404
    return resp


# load model
svm_model = pickle.load(open('svm_best.pkl', 'rb'))

# load transormer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    feature_transformer = pickle.load(f)


@application.route('/')
def index():

    return "Detect Emotions from Text"

@app.route('/predict_emotions', methods=['GET'])
def predict_emotions():
    args = request.args
    logging.info('args:%s' % args)
    text = args.get('text', default = None, type = str)
    
    if text is not None:
        text_features = feature_transformer.transform(np.array([text]))
        label = svm_model.predict(text_features)
    else:
        return 'Error:  %s'%(text)
    
    result = {'text':text, 'emotion':label}
    
    with open('result.json', 'w') as f:
        json.dump(result, f)

    return result



if __name__ == '__main__':
    application.run(debug=True, use_reloader=True)