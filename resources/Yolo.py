# Copyright © 2020 BYTEPAL AI, LLC And Its Affiliates. All rights reserved.

from flask_restful import Resource
from flask import Flask, request, Response, jsonify
import jsonpickle
import base64
import cv2
import numpy as np
from PIL import Image
import subprocess
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import time
from Prediction import *


class Yolo(Resource):
    def get(self):
        return {"message": "Hello, World Object Detection!"}

    def post(self): # Post route, will need better names
        global model
        r = request
        print("stuck here", request, request.files['image'])
        img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
	    #print("getting here)
        print("getting here")
        # Save image
        cv2.imwrite('./image.jpg', img) # Do i need to always save and read, maybe that might waste some time, also that might be customer data so maybe i shouldn't save or maybe i should
        # I guess i might to save at least for saving a database for the customer
        # To be reviewed, but i might not be an issue it just takes like 0.04 seconds
        start = time.time() # Time Started
        prediction, output = predict_yolo('./image.jpg', model)
        #print(output)
        cv2.imwrite('./predictions.jpg', prediction) # Do i need to always save first, let me fix it
        print("it took", time.time() - start, "seconds.")
        # It takes about 9 seconds on my old mac - should way faster on a compute shape on digital Ocean
        # do some fancy processing here....
        data = open('./predictions.jpg', 'rb').read()
        #response = {"image":data, "labels":output}
        response = {"labels":output}
        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response) # Just encoding format using jsonpickle
        return Response(response=response_pickled, status=200, mimetype="application/json") # Send the image annotedt to the client
