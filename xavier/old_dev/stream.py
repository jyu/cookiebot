import asyncio
import websockets
import numpy as np
import cv2
import socket
import base64
import sys

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

async def image(ws, path):
    while True:
        jpg_as_text = await ws.recv()
        print("Got image")
        img = decode_img(jpg_as_text)
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop([datum])
        img = datum.cvOutputData
        text = encode_img(img)
        await ws.send(text)


host_name = socket.gethostbyname(socket.gethostname())
start_server = websockets.serve(image, host_name, 5000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
    
