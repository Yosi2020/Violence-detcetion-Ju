from tensorflow.keras.models import load_model
from collections import deque
from imutils.video import VideoStream
from imutils.video import FPS
from my_lib.mailer import Mailer
from my_lib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, cv2, datetime, os
from itertools import zip_longest

os.chdir(r"C:\Users\Eyosiyas\Desktop\Final code")
CLASSES = ["Non_violence","Violence"]

conf = {"input":r"E:\My project\Real Life Violence Dataset\Violence\V_50.mp4", 
#"model_input" : r"C:\Users\Eyosiyas\Desktop\Food-11\best.model",
"model_input" : "final_v_pg3.model", 
"output" : "output", 
"size" : 128}

#loading the train model
print("[INFO] loading model ...")
model = load_model(conf["model_input"])
#lb = pickle.loads(open("le.cpickle", "rb").read())

# if a video path was not supplied, grab a reference to the ip camera
if not conf.get("input", False):
    print("[INFO] Starting the live stream..")
    vs = VideoStream(config.url).start()
    time.sleep(2.0)
else:
    # otherwise, grab a reference to the video file
    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    vs = cv2.VideoCapture(conf["input"])

# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=conf["size"])

# start the frames per second throughput estimator
fps = FPS().start()

if config.Thread:
    vs = thread.ThreadingClass(config.url)

writer = None
(W, H) = (None, None)
# loop over frames from the video file stream
while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1] if conf.get("input", False) else frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if conf["input"] is not None and frame is None:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # clone the output frame, then convert it from BGR to RGB
    # ordering, resize the frame to a fixed 224x224, and then
    # perform mean subtraction
    output = frame.copy()
    output = imutils.resize(output, width=400)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    #frame = frame.reshape(224, 224, 3)/255
    frame = frame - mean
    # make predictions on the frame and then update the predictions
    # queue
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)
    print(preds)
    # perform prediction averaging over the current history of
    # previous predictions
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    print(i)
    label = CLASSES[i]
    # draw the activity on the output frame
    text = "{}".format(label)
    if label == "Violence":
        cv2.putText(output, text, (3, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1.25, (0, 0, 255), 3)
    else:
        cv2.putText(output, text, (3, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1.25, (0, 255, 0), 3)
    
    # if violence is detected send email code...
    if label == "Violence":
        cv2.putText(output, "-ALERT: Violence....", (10, frame.shape[0] - 80),
            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)

        if config.ALERT:
            print("[INFO] Sending email alert..")
            Mailer().send(config.MAIL)
            print("[INFO] Alert sent")
    

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(conf["output"], fourcc, 30,
            (W, H), True)
    # write the output frame to disk
    writer.write(output)
    # show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()