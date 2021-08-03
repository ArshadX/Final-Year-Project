from flask import Flask, render_template, request,jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re

img_size=100

app = Flask(__name__) 

model=load_model('model/model-015.model')

label_dict={0:'Covid19 Negative', 1:'Covid19 Positive'}

def preprocess(img):

	img=np.array(img)
	if(img.ndim==3):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray=img

	resized=cv2.resize(gray,(img_size,img_size))
	
	gray=gray/255.0
	# reshaped=np.reshape(resized,(resized.shape[0],img_size,img_size,1))
	reshaped=resized.reshape(1,img_size,img_size)
	# data=np.array(data)/255.0
	# data=np.reshape(data,(data.shape[0],img_size,img_size,1))
	# target=np.array(target)
	#NEW LINE
	# reshaped=np.expand_dims(reshaped,axis=0)
	return reshaped

@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)

	prediction = model.predict(test_image)
	result=np.argmax(prediction,axis=1)[0]
	accuracy=float(np.max(prediction,axis=1)[0])

	label=label_dict[result]

	print(prediction,result,accuracy)

	response = {'prediction': {'result': label,'accuracy': accuracy}}

	return jsonify(response)

app.run(debug=True)

#<img src="" id="img" crossorigin="anonymous" width="400" alt="Image preview...">
