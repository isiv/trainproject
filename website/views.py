from flask import Flask, flash, request, redirect, url_for, render_template, Blueprint
from werkzeug.utils import secure_filename, os

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

import os
import numpy as np
import pandas as pd

from PIL import Image

## Define static variables and paths
UPLOAD_FOLDER = 'website/static/uploads/'

## Define the blueprint
views = Blueprint('views', __name__)

## Load the model.
model = MobileNetV2(weights='imagenet')


## Define the allowed file extensions
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_filename(name):
    return '.' in name and name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


## Define the route for the home page
@views.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@views.route('/', methods=['POST'])
def upload_image():

    if 'file' not in request.files:
        flash('No file was uploaded.')
        return redirect(request.url) 
   
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_filename(file.filename):

        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))

        img = image.load_img(os.path.join(UPLOAD_FOLDER, filename), target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        preds = model.predict(img)
        predictions = decode_predictions(preds, top=5)[0]
        predictions_filtered = pd.DataFrame.from_records(predictions, columns =['class_name', 'class_description', 'score']).query("class_description in ('bullet_train', 'electric_locomotive','freight_car', 'steam_locomotive')")
        
        if predictions_filtered.size == 0:
            flash('This is not a train 😒')
            return render_template('index.html', filename=filename)
            #return str(predictions)
        else:
            score = predictions_filtered['score'].iloc[0]
            flash('It is a train! 😍')
            return render_template('index.html', filename=filename)

    else:
        flash('Not an allowed file type, please use png, jpg, jpeg, gif.')
        return redirect(request.url)


@views.route('/display/<filename>', methods=['GET'])
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

