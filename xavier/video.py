import sys
import cv2
import os
import argparse

# Location of OpenPose python binaries
openpose_path = '../../openpose'
openpose_python_path = openpose_path + '/build/python'
sys.path.append(openpose_python_path)

from openpose import pyopenpose as op

params = dict()
params["model_folder"] = openpose_path + "/models/"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

img_path = openpose_path + "/examples/media/COCO_val2014_000000000192.jpg"

# Process Image
datum = op.Datum()
imageToProcess = cv2.imread(img_path)
print(imageToProcess)
print(imageToProcess.shape)


datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop([datum])

print("Body keypoints: \n" + str(datum.poseKeypoints))
