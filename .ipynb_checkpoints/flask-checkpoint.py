import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import json
import time
import _pickle as pickle
from tqdm import tqdm
from PIL import Image
import pandas as pd
import random
from IPython.display import Video
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
import sys
import statistics
import pickle
from statistics import mode
from tensorflow.keras.utils import to_categorical

from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input as pre

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer
from sklearn.preprocessing import OneHotEncoder

from pathlib import Path

import tensorflow as tf
from keras.layers import Input

from google.cloud import storage
from google.cloud.storage import blob
import keras

from mtcnn.mtcnn import MTCNN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, url_for, render_template, flash, redirect
import uuid
import os
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNet
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions
#from app import app

LABELS = ['FAKE', 'REAL']
IMG_SIZE = 299
def build_feature_extractor():
    feature_extractor = InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()

import sys, os

# original
def extract_face(frame,detector,required_size=(299, 299)):
        # create the detector, using default wei        ghts
        # detect faces in the image
        faces = detector.detect_faces(frame)
        # extract the bounding box from the first face
        face_images = []
        arr =[]
        for face in faces:
            if face["confidence"] > 0.96:
                arr.append(face["confidence"])
                # extract the bounding box from the requested face
                x1, y1, width, height = face['box']
                x2, y2 = x1 + width, y1 + height
                
                # extract the face
                face_boundary = frame[y1:y2, x1:x2]

                # resize pixels to the model size
                face_image = Image.fromarray(face_boundary)
                face_image = face_image.resize(required_size)
                face_array = np.asarray(face_image)
                face_images.append(face_array)

        return face_images

def get_embedding(face,model):
    # convert into an array of samples
    sample = [np.asarray(face, 'float32')]
    # prepare the face for the model, e.g. center pixels
    sample = pre(sample, version=2)
    # perform prediction
    yhat = model.predict(sample)    
    return yhat       


def is_match(i,j,a,b,show_faces,n_faces,embedings,thresh=0.4):
    # calculate distance between embeddings
    score = cosine(embedings[i][j], embedings[a][b])
    if show_faces==True:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(n_faces[i][j])
        ax1.set_title('ID face')
        ax2.imshow(n_faces[a][b])
        ax2.set_title('Subject face')

    return score

detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(299, 299, 3), pooling='avg')

def nueva(path):
    sys.stdout = open(os.devnull, 'w')
    video = "static/uploads/" + path
    print(path)
    v_cap = cv2.VideoCapture(video)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through video, taking a handful of frames to form a batch
    faces = []
    modas = {}
    moda = 0
    for i in range(v_len):
        # Load face
        success = v_cap.grab()
        if moda < 50:
            success, frame = v_cap.retrieve()
            if success:
                cara = extract_face(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),detector)
                if cara != []:
                    tam = len(cara)
                    modas[tam] = modas.get(tam, 0) +1
                    faces.append(cara)
                    moda = max(moda, modas[tam])
                else:
                    continue
            else:
                continue

    caras = [len(x) for x in faces]
    mode = statistics.mode(caras)
    n_faces = [x for x in faces if len(x) == mode]

    emb = [[get_embedding(y,model) for y in x] for x in n_faces]
    sys.stdout = sys.__stdout__
    for i in range(1,len(emb)): # frame actual
        for j in range(len(emb[i])): # frame anteriorx  
            s = 0.4
            p = 0
            for k in range(len(emb[i-1])): # cara actual
                if is_match(i,j,i-1,k,False,n_faces,emb) < s:
                    s = is_match(i,j,i-1,k,False,n_faces,emb)
                    p = k
            n_faces[i][j],n_faces[i][p] = n_faces[i][p],n_faces[i][j] 
            emb[i][j], emb[i][p] = emb[i][p], emb[i][j]
    
    return n_faces

def prediccion(path):
    arr = nueva(path)
    a = {}
    for i in range(len(arr[0])):
        temp = list(map(lambda x: x[i] , arr))
        img_path = str(i)+path.split(".")[0]+".jpg"
        cv2.imwrite("static/uploads/"+img_path, cv2.cvtColor(temp[0], cv2.COLOR_RGB2BGR))
        temp = np.array(temp)
        mapa = feature_extractor.predict(temp)

        cnnrnn = tf.keras.models.load_model("static/deepfake_rnn_3.h5")
        prediccion = cnnrnn.predict(mapa[None,:,:])
        a[img_path] = "El rostro es :{} \n Con una probabilidad de: {:.2%}".format(LABELS[np.argmax(prediccion)], np.amax(prediccion) )
    return a
    
ALLOWED_EXTENSION = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4'])
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

def allowed_file(filename):
    return '.' in filename and \
     filename.rsplit('.',1)[1] in ALLOWED_EXTENSION

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def upload_form():
    return render_template('upload.html', prediction={})

@app.route('/', methods=['POST'])
def upload_video():
    prediction={}
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')
        pred = prediccion(file.filename)
        return render_template('upload.html', filename=filename, prediction=pred)
    
    
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

@app.route('/display/<filename>')
def display_video(filename):
    #print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()