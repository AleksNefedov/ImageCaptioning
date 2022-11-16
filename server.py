# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 15:38:35 2022

@author: Aleksey Nefedov
"""

import os
import torch
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from celery import Celery
from celery.result import AsyncResult
import matplotlib.pyplot as plt
from skimage.transform import resize
from libs.caption_generation import beheaded_inception_v3, n_tokens, CaptionNet, generate_caption, load_model
import time
import json

# create web server and broker for asynchronous tasks
celery_broker = os.environ.get('CELERY_BROKER', 'localhost')
celery_backend = os.environ.get('CELERY_BACKEND', 'localhost')
celery_app = Celery('server', broker=celery_broker, backend=celery_backend)
server = Flask(__name__)

# load model
try:
    mlflow_host = os.environ.get('MLFLOW_HOST', 'localhost')
    mlflow_port = os.environ.get('MLFLOW_PORT')
    mlflow_url = f'http://{mlflow_host}:{mlflow_port}'
    experiment_name = 'model_run'
    model = load_model(mlflow_url, experiment_name)
except:
    checkpoint_path = './checkpoints/best_model.pkl'
    model = CaptionNet(n_tokens)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['best_weights'])	    
finally:
    model.eval()

# load vectorizer
vectorizer = beheaded_inception_v3().eval()


@celery_app.task
def get_caption(img_path):    
    img = plt.imread(img_path)
    img = resize(img, (299, 299))
    caption = generate_caption(model, vectorizer, img, t=5.)
    return str(caption)

@server.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@server.route('/caption', methods=['GET', 'POST'])
def caption_handler():
    if request.method == 'POST':
        # get file from post request
        f = request.files['file']

        # save file to ./images
        par_path = os.path.dirname(__file__)
        img_path = os.path.join(
            par_path, 'images', secure_filename(f.filename)
        )
        f.save(img_path)

        # launch celery task and get the task id
        task = get_caption.delay(img_path)
        response = {'task_id': task.id}

        return json.dumps(response)
        

@server.route('/caption/<task_id>')
def caption_check_handler(task_id):
    task = AsyncResult(task_id, app=celery_app)
    response = {
        'ready': task.ready(),
        'result': str(task.result) if task.ready() else None
    }
    print(task.ready(), str(task.result))
    return json.dumps(response)


if __name__ == '__main__':    	
    server.run(host='0.0.0.0', port=5000)
