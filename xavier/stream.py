# Streaming point
import asyncio
import websockets
import numpy as np
import cv2
import socket
import base64
import sys
import math

from sklearn import preprocessing
from get_video_feat import getKeyPointsFeat, getKeyPointsNewFeat, getKeyPointsLocationFeat
import os

import tensorflow as tf
from tensorflow.keras.models import load_model

my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

point_nn = load_model("prod_models/loc_reg_point_nn_2.h5")
classes_x = [-2, -1, 0, 1, 2]
classes_y = [0, 1, 2]

# Location of OpenPose python binaries
#openpose_path = "usr/lib/openpose"
openpose_path = "../../openpose"
openpose_python_path = openpose_path + "/build/python"
sys.path.append(openpose_python_path)

from openpose import pyopenpose as op

# OpenPose params
params = dict()
params["model_folder"] = openpose_path + "/models/"
params["tracking"] = 5
params["number_people_max"] = 1 

# Start OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def encode_img(img):
    _, buf = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buf)
    return jpg_as_text

def decode_img(text):
    buf = base64.b64decode(text)
    img = cv2.imdecode(np.fromstring(buf, dtype=np.uint8), -1)
    return img

def keypointsToPosition(keypoints, img_shape):
    if len(keypoints.shape) == 0:
        return ""
    feat = getKeyPointsNewFeat(keypoints)
    if np.sum(feat) == 0:
        return ""
    feat = preprocessing.scale(feat)
    feat = feat.reshape(1, -1)

    loc = getKeyPointsLocationFeat(keypoints, img_shape)
    loc = loc.reshape(1, -1)

    out = point_nn.predict([feat, loc])
    out_x = int(round(out[0][0][0]))
    out_y = int(round(out[1][0][0]))
    out_x = min(len(classes_x) - 1, out_x)
    out_y = min(len(classes_y) - 1, out_y)
    print(out_x, out_y)
    x = classes_x[out_x]
    y = classes_y[out_y]
    return (x,y)

def getPointTriggerDist(keypoints):
    if len(keypoints.shape) == 0:
        return None
    person = keypoints[0]
    x = 0
    y = 1

    r_elbow = person[3]
    r_wrist = person[4]
    l_elbow = person[6]
    l_wrist = person[7]
    
    r_point_dist = math.sqrt((l_elbow[x] - r_wrist[x]) ** 2 + (l_elbow[y] -
            r_wrist[y]) ** 2)
    l_point_dist = math.sqrt((r_elbow[x] - l_wrist[x]) ** 2 + (r_elbow[y] -
            l_wrist[y]) ** 2)


    return round(r_point_dist), round(l_point_dist)

async def image(ws, path):
    while True:
        jpg_as_text = await ws.recv()
        print("Got image")
        img = decode_img(jpg_as_text)
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop([datum])
        img = datum.cvOutputData
        pos = keypointsToPosition(datum.poseKeypoints, img.shape)
        #text = encode_img(img)
        r_point_dist, l_point_dist = getPointTriggerDist(datum.poseKeypoints)
        await ws.send(str(pos) + '_' + str(r_point_dist) + '_' + str(l_point_dist))


host_name = socket.gethostbyname(socket.gethostname())
start_server = websockets.serve(image, host_name, 5000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
    
