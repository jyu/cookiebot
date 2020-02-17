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

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

img_path = openpose_path + "/examples/media/COCO_val2014_000000000192.jpg"

# Process Image, expects cv2 image
def processImage(img):
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop([datum])
    cv2.imwrite("out.jpg", datum.cvOutputData)
    return datum.poseKeypoints

imageToProcess = cv2.imread(img_path)
print("Image shape", imageToProcess.shape)
start = time.time()
keypoints = processImage(imageToProcess)
end = time.time()
#print("Body keypoints: \n" + str(keypoints))
print("Total time: " + str(end - start))
