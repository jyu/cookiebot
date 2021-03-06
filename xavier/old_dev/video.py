import sys
import cv2
import os
import argparse
import time

# Location of OpenPose python binaries
openpose_path = "../../openpose"
openpose_python_path = openpose_path + "/build/python"
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

# Process Image, expects cv2 image
def processImage(img, i):
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop([datum])
    if i != -1:
        cv2.imwrite("out/out" + str(i) + ".jpg", datum.cvOutputData)
    return datum.poseKeypoints


def keypointsToCommand(keypoints):
    person = keypoints[0]
    # For getting the index of the parts we want
    #poseModel = op.PoseModel.BODY_25
    #print(op.getPoseBodyPartMapping(poseModel))

    # Result is [x, y, confidence]
    x = 0
    y = 1
    res = {
        "r_shoulder": person[2],
        "r_elbow": person[3],
        "r_wrist": person[4],
        "l_shoulder": person[5],
        "l_elbow": person[6],
        "l_wrist": person[7],
    }
    
    #print(res)
    right_hand_raised = False
    left_hand_raised = False
    
    # Raised arms
    raised_arm_delta = 30
    if (res["r_wrist"][y] + raised_arm_delta < res["r_shoulder"][y] and 
        res["r_elbow"][y] + raised_arm_delta < res["r_shoulder"][y]):
        right_hand_raised = True
    if (res["l_wrist"][y] + raised_arm_delta < res["l_shoulder"][y] and 
        res["l_elbow"][y] + raised_arm_delta < res["l_shoulder"][y]):
        left_hand_raised = True

    # Teleop    
    teleop_mode = False
    """
    delta_tolerance = 30
    # Teleop mode: when right elbow is raised
    if abs(res["r_elbow"][y] - res["r_shoulder"][y]) < delta_tolerance:
        teleop_mode = True
    """

        
    if right_hand_raised and left_hand_raised:
        return "both_hands_raised"
    if right_hand_raised:
        return "right_hand_raised"
    if left_hand_raised:
        return "left_hand_raised"
    if teleop_mode:
        return "teleop_mode"

"""
# For videos
start = time.time()
#vidcap = cv2.VideoCapture("data/rama_teleop.mp4")
vidcap = cv2.VideoCapture("data/rama.mp4")
success, img = vidcap.read()
print("Image shape", img.shape)
count = 0
while success:
    count += 1
    keypoints = processImage(img, count)
    command = keypointsToCommand(keypoints)
    if (command != None):
        print(command, count)
    success, img = vidcap.read()
    
end = time.time()
total_time = end - start

print("Total time:     ", total_time)
print("FPS:            ", count / total_time)
print("Time per frame: ", total_time / count)

"""
# For images
img_paths = [
    #"data/rama.jpg", 
    #"data/back.jpg", 
    #"data/front.jpg", 
    # "data/frontview.jpg", 
    # "data/sideview.jpg", 
    "data/overhead.jpg",
]

j = 0
for img_path in img_paths:
    img = cv2.imread(img_path)
    print("Image shape", img.shape)
    print("Image path", img_path)
    for i in range(1):
        start = time.time()
        keypoints = processImage(img, j)
        end = time.time()
        #print("Body keypoints: \n" + str(keypoints))
        print("Command", keypointsToCommand(keypoints))
        print("Total time: " + str(end - start))    

    j += 1
