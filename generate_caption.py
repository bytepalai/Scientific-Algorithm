# Copyright © 2020 BYTEPAL AI, LLC And Its Affiliates. All rights reserved.

from pickle import load
from numpy import argmax
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

def extract_features(filename):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = model.predict(image, verbose=0)
	return feature

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def output_desc(in_text):
    output = in_text
    output = output.split('startseq')
    output = output[1]
    output = output.split("endseq")
    output = output[0]

    return output[1:]

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break

    return output_desc(in_text)

#parser=argparse.ArgumentParser()
#parser.add_argument('filename')
#args=parser.parse_args()
#model_cnn = VGG16()
tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 34
final_model = load_model('model_12.h5')

def caption(filename):
    global final_model
    global tokenizer
    global max_length
    photo = extract_features(filename)
    description = generate_desc(final_model, tokenizer, photo, max_length)
    description = {'description':description}
    return description

#description = caption('image2.jpg')
#print(description)
