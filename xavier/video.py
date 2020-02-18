import sys
import cv2
import os
import argparse
import time

# Location of OpenPose python binaries
openpose_path = '../../openpose'
openpose_python_path = openpose_path + '/build/python'
sys.path.append(openpose_python_path)

from openpose import pyopenpose as op

# Initialize OpenPose
# TODO: tune flags to optimize speed of openpose
params = dict()
params["model_folder"] = openpose_path + "/models/"
params["tracking"] = 5
params["number_people_max"] = 1

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

img_paths = [
    "data/rama.jpg", 
]

# Process Image, expects cv2 image
def processImage(img, i):
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop([datum])
    if i != -1:
        cv2.imwrite("out/out" + str(i) + ".jpg", datum.cvOutputData)
    return datum.poseKeypoints

"""
# For videos
start = time.time()
vidcap = cv2.VideoCapture('data/rama.mp4')
success, img = vidcap.read()
print("Image shape", img.shape)
count = 0
while success:
    count += 1
    processImage(img, count)

    if count % 200 == 0:
        print("Done processing frame:", count)
        print("Time so far:", time.time() - start)
    success, img = vidcap.read()
    
end = time.time()
total_time = end - start

print("Total time:     ", total_time)
print("FPS:            ", count / total_time)
print("Time per frame: ", total_time / count)

"""

# For images
for img_path in img_paths:
    img = cv2.imread(img_path)
    print("Image shape", img.shape)
    for i in range(1):
        start = time.time()
        keypoints = processImage(img, -1)
        end = time.time()
        print("Body keypoints: \n" + str(keypoints))
        print(len(keypoints[0]))
        print("Total time: " + str(end - start))

poseModel = op.PoseModel.BODY_25
print(op.getPoseBodyPartMapping(poseModel))
