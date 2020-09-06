# Copyright © 2020 BYTEPAL AI, LLC And Its Affiliates. All rights reserved.

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import time

def predict_imagenet(img_path, model):
    # Process the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Prediction
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    predictions = decode_predictions(preds, top=3)[0]
    output = {"labels":[]}
    # Building the output Format

    for object in predictions:
        current = {"label":object[1], "confidence":float(object[2]*100)}
        output["labels"].append(current)
    # Output format
    # {'labels': [{'label': 'bikini', 'confidence': 30.677080154418945}, {'label': 'maillot', 'confidence': 9.665444493293762}, {'label': 'sandbar', 'confidence': 5.5834464728832245}]}
    return output

model = ResNet50(weights='imagenet')
