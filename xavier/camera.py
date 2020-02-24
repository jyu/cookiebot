import cv2
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np

import time

time.sleep(5)

# Location of OpenPose python binaries
openpose_path = "../../openpose"
openpose_python_path = openpose_path + "/build/python"
sys.path.append(openpose_python_path)

from openpose import pyopenpose as op

params = dict()
params["model_folder"] = openpose_path + "/models/"
params["tracking"] = 5
params["number_people_max"] = 1 

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 900, 1000)
success, img = cap.read()
print(img.shape)

# sliding window for timing data
window_size = 100
window = []

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
    delta_tolerance = 30

    # Teleop mode: when right elbow is raised
    if abs(res["r_elbow"][y] - res["r_shoulder"][y]) < delta_tolerance:
        teleop_mode = True
        
        # Check flip, user turned around
        flip = False
        if res["r_shoulder"][x] - res["l_shoulder"][x] > 0:
            flip = True
            print("flipping")
        
        if res["r_elbow"][x] - res["r_shoulder"][x] < -delta_tolerance:
            teleop_command = "teleop_right"
            if flip:
                teleop_command = "teleop_left"
        elif res["r_elbow"][x] - res["r_shoulder"][x] > delta_tolerance:
            teleop_command = "teleop_left"
            if flip:
                teleop_command = "teleop_right"
        else:
            teleop_command = "teleop_straight"

        
    if right_hand_raised and left_hand_raised:
        return "both_hands_raised"
    if right_hand_raised:
        return "right_hand_raised"
    if left_hand_raised:
        return "left_hand_raised"
    if teleop_mode:
        return teleop_command
    return "no command"

success = True
while success:
    start_time = time.time()
    success, img = cap.read()
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop([datum])
    img = datum.cvOutputData
    img = np.array(img, dtype=np.uint8)
    command = keypointsToCommand(datum.poseKeypoints)
    end_time = time.time()

    window.append(end_time - start_time)
    if len(window) > window_size:
        window.pop(0)
    sec = sum(window) / len(window)
    ms = round(sec * 1000, 3)
    fps = round(1 / sec, 3)

    print(ms, "ms,", fps, "fps", end="\r")

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(fps) + " FPS", (20, 20), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, str(ms) + " ms per frame", (20, 50), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, str(command), (20, 100), font, .5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('image',img)
    cv2.waitKey(1)
