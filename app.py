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
import sqlite3
from flask import Flask, redirect, url_for, request, render_template,g
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


def connect_db():
    sql = sqlite3.connect('data.db')
    sql.row_factory = sqlite3.Row
    return sql

def get_db():
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db

@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()

@app.route('/data')
def data():
    db=get_db()
    cur=db.execute('select username, phone, result from data')
    results=cur.fetchall()
    return render_template('database.html', results=results)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
        if request.method == 'POST':
            # Get the file from post request
            if 'file' not in request.files:
                flash('No file part')
                return redirect(index.url)
            f = request.files['file']
            if f.filename == '':
                flash('No selected file')
                return redirect(index.url)

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            name=request.form['name']
            phone=request.form['phone']
            
            
            # Make prediction
            preds = model_predict(file_path, model)
            os.remove(file_path)#removes file from the server after prediction has been returned
            result= False
            str1 = 'Covid19 Negative'
            str2 = 'Covid19 Positive'
            if preds == 1:
                result= False
            else:
                result= True
            db=get_db()
            db.execute('insert into data (username, phone, result) values(?,?,?)',[name,phone,result])
            db.commit()
            
            if preds == 1:
                return render_template('result.html', name=request.form['name'], phone=request.form['phone'], result=str1)
            else:
                return render_template('result.html', name=request.form['name'], phone=request.form['phone'], result=str2)
        return None

if __name__ == '__main__':
        app.run(threaded = False)

