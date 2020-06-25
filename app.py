from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.models import load_model
from keras.preprocessing import image
import keras.backend.tensorflow_backend as tb

tb._SYMBOLIC_SCOPE.value = True

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/Covid.h5'

#Load your trained model

model = load_model(MODEL_PATH)
model._make_predict_function()



print('Model loaded. Start serving...')



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

   
    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
        if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            preds = model_predict(file_path, model)
            os.remove(file_path)#removes file from the server after prediction has been returned
        
            str1 = 'Normal'
            str2 = 'Covid19 Positive'
            
            if preds == 1:
                return str1
            else:
                return str2
        return None

if __name__ == '__main__':
        app.run(threaded = False)

