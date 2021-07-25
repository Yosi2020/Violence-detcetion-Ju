# import the necessary packages
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os

os.chdir(r"C:\Users\Eyosiyas\Desktop\Final code")
CLASSES = ["Non_Violence", "Violence"]

# load the input image and then clone it so we can draw on it later
image = cv2.imread(r"C:\Users\Eyosiyas\Desktop\Violence detection\Dataset\train\Non_Violence\NV_33 27.jpg")
output = image.copy()
output = imutils.resize(output, width=500)
# our model was trained on RGB ordered images but OpenCV represents
# images in BGR order, so swap the channels, and then resize to
# 224x224 (the input dimensions for VGG16)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
# convert the image to a floating point data type and perform mean
# subtraction
image = image.astype("float32")
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
image = image - mean

# load the trained model from disk
print("[INFO] loading model...")
model = load_model('final_V_pg3.model')
# pass the image through the network to obtain our predictions
preds = model.predict(np.expand_dims(image, axis=0))[0]
print(preds)
i = np.argmax(preds)
label = CLASSES[i]
print(label)
# draw the prediction on the output image
text = "{}: {:.2f}%".format(label, preds[i] * 100)
cv2.putText(output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
	(0, 255, 225), 2)
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)