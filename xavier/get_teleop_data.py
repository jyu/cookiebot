import cv2
import subprocess
import sys
import numpy as np
import time
from websocket import create_connection
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', action="store_true")
parser.add_argument('-d', action="store_true")
parser.add_argument('-mode')

args = parser.parse_args()
use_server = args.s
display = args.d
mode = args.mode
print("COLLECTING DATA FOR MODE", mode)

if use_server:
    # Use websockets
    ws = create_connection("ws://localhost:5000/gestures")

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

# Start reading camera feed
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L)
#cap = cv2.VideoCapture("/dev/video0", cv2.CAP_GSTREAMER)

if display:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 900, 1000)

success, img = cap.read()
print(img.shape)

# sliding window for timing data
window_size = 100
window = []

# Last command 
last_command = "none"

# Hold gesture for gesture_buffer frames before changing the command
gesture_buffer = 10
new_command = "none"
new_command_count = 0

def isConfidentAboutArm(res, confidence_threshold, side):
    conf = 2
    if side == "right":
        return (
            res["r_wrist"][conf] > confidence_threshold and
            res["r_elbow"][conf] > confidence_threshold and
            res["r_shoulder"][conf] > confidence_threshold
        )
    elif side == "left":
        return (
            res["l_wrist"][conf] > confidence_threshold and
            res["l_elbow"][conf] > confidence_threshold and
            res["l_shoulder"][conf] > confidence_threshold
        )
    return False

def getKeyPointsFeat(keypoints):
    if len(keypoints.shape) == 0:
        return None
    person = keypoints[0]
    x = 0
    y = 1
    chest = person[1]

    # Part indicies
    # R shoulder, R elbow, R wrist, L shoulder, L elbow, L wrist
    # Midhip, RHip, LHip, Reye, LEye
    parts = [2,3,4,5,6,7,8,9,12,15,16]
    
    feat = []
    for p in parts:
        feat.append(chest[x] - person[p][x]) 
        feat.append(chest[y] - person[p][y])

    return np.array(feat)

def keypointsToCommand(keypoints):
    if len(keypoints.shape) == 0:
        return "none"
    person = keypoints[0]
    # For getting the index of the parts we want
    #poseModel = op.PoseModel.BODY_25
    #print(op.getPoseBodyPartMapping(poseModel))

    # Result is [x, y, confidence]
    x = 0
    y = 1
    conf = 2
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
    confidence_threshold = 0.1
    
    # Raised arms
    raised_arm_delta = 30
    if (res["r_wrist"][y] + raised_arm_delta < res["r_shoulder"][y] and 
        res["r_elbow"][y] + raised_arm_delta < res["r_shoulder"][y] and
        isConfidentAboutArm(res, confidence_threshold, "right")):
        right_hand_raised = True
    if (res["l_wrist"][y] + raised_arm_delta < res["l_shoulder"][y] and 
        res["l_elbow"][y] + raised_arm_delta < res["l_shoulder"][y] and
        isConfidentAboutArm(res, confidence_threshold, "left")):
        left_hand_raised = True

    # Teleop    
    teleop_mode = False
    delta_tolerance = 30

    # Teleop mode: when right elbow is raised, and wrist is not too high above
    # elbow
    if (abs(res["r_elbow"][y] - res["r_shoulder"][y]) < delta_tolerance and
        abs(res["r_wrist"][y] - res["r_elbow"][y]) < delta_tolerance * 2):

        teleop_mode = True
        
        # Check flip, user turned around
        flip = False
        if res["r_shoulder"][x] - res["l_shoulder"][x] > 0:
            flip = True
            #print("flipping")
        
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
        return "stop"
    if right_hand_raised:
        return "to_me"
    if left_hand_raised:
        return "go_home"
    if teleop_mode:
        return teleop_command
    return "none"

frames = 0
success = True
written = 0
while success:
    frames += 1
    start_time = time.time()
    success, img = cap.read()
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop([datum])
    img = datum.cvOutputData
    img = np.array(img, dtype=np.uint8)
    keypoints = datum.poseKeypoints
    command = keypointsToCommand(datum.poseKeypoints)

    # We only want to change teleop command if we saw it 3 frames in a row
    if command != last_command:

        # We have never seen this command before 
        if (command != new_command and 
            not "teleop" in last_command):

            new_command = command
            new_command_count = 1
        else:
            new_command_count += 1
            new_command = command
            # We saw it enough times, use it
            if new_command_count > gesture_buffer:
                last_command = command
                # Communicate to server
                if use_server:
                    print("sending command", command)
                    ws.send(command)
                print("new command", command)

    end_time = time.time()

    window.append(end_time - start_time)
    if len(window) > window_size:
        window.pop(0)
    sec = sum(window) / len(window)
    ms = round(sec * 1000, 3)
    fps = round(1 / sec, 3)

    #print(ms, "ms,", fps, "fps", end="\r")
    #print(command, end="\r")
    
    if display:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(fps) + " FPS", (20, 20), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(ms) + " ms per frame", (20, 50), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(command), (20, 100), font, .5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('image',img)
        key = cv2.waitKey(1)

        # Auto method

        # Save as feat only on 5th frame
        if frames % 5 != 0:
            continue

        # Must follow teleop heuristic
        if not "teleop" in command:
            continue

        feat = getKeyPointsFeat(keypoints)

        if feat is None:
            continue

        print(feat)
        print(feat.shape)
        folder = "teleop_data"
        f = mode
        fwrite = open(folder + "/" + f, 'a')
        line = str(feat[0])
        for m in range(1, feat.shape[0]):
            line += ";" + str(feat[m])
        line += "\n"
        fwrite.write(line)
        fwrite.close()
        written += 1
        print("written", written)

        # Key method MANUAL
        """
        if key != -1:
            # Must follow teleop heuristic
            if not "teleop" in command:
                continue

            feat = getKeyPointsFeat(keypoints)

            if feat is None:
                continue

            if not key in [81, 82, 83]:
                continue

            print(feat)
            print(feat.shape)
            folder = "straight"
            if key == 81:
                f = "left"  
            if key == 82:
                f = "straight"
            if key == 83:
                f = "right"
            fwrite = open(folder + "/" + f)
            line = str(feat[0])
            for m in range(1, feat.shape[1]):
                line += ";" + str(feat[m])
            fwrite.write(line)
            fwrite.close()
        """
            
