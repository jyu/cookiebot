import cv2
import subprocess
import sys
import numpy as np
import time
from websocket import create_connection
import argparse
import pickle
from sklearn import preprocessing
from get_gesture_data import getKeyPointsFeat
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', action="store_true") # Use server
parser.add_argument('-d', action="store_true") # Use display
parser.add_argument('-v') # Use video

args = parser.parse_args()
use_server = args.s
display = args.d
video = args.v

if use_server:
    # Use websockets
    ws = create_connection("ws://localhost:5000/gestures")

# Location of SVM
teleop_svm = pickle.load(open("prod_models/teleop.svm", "rb"))
#point_svm = pickle.load(open("prod_models/point.svm", "rb"))
point_svms = {}
point_classes = os.listdir("point_data")
for m in os.listdir("models"):
    if not "point" in m:
        continue    

    i = int(m.replace("point_", "").replace(".svm",""))
    point_svms[i] = pickle.load(open("models/" + m, "rb"))

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
cap = None
print("video", video)
if video:
    path = "data/" + video
    print("Using video as input from path", path)
    cap = cv2.VideoCapture(path)
    out_f = "out/" + video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_f, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
else:    
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L)

if display:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 900, 1000)


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

def keypointsToPosition(keypoints):
    if len(keypoints.shape) == 0:
        return ""
    feat = getKeyPointsFeat(keypoints)
    if np.sum(feat) == 0:
        return ""
    feat = preprocessing.scale(feat)
    feat = feat.reshape(1, -1)
    
    max_i_pred = (-1, -1)
    for m in point_svms.keys():
        point_svm = point_svms[m]
        pred = point_svm.predict(feat)
        if max_i_pred[0] == -1 or pred > max_i_pred[1]:
            max_i_pred = (m, pred)

    return point_classes[max_i_pred[0]]

def keypointsToTeleop(keypoints):
    if len(keypoints.shape) == 0:
        return ""
    feat = getKeyPointsFeat(keypoints)
    if np.sum(feat) == 0:
        return ""
    feat = preprocessing.scale(feat)
    feat = feat.reshape(1, -1)
    svm_res = teleop_svm.predict(feat)
    commands = ["teleop_left", "teleop_straight", "teleop_right"]
    return commands[svm_res[0]]

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

        # NO MODEL METHOD
        # Check flip, user turned around
        flip = False
        if res["r_shoulder"][x] - res["l_shoulder"][x] > 0:
            flip = True
        
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

def displayText(image, text, position):
    white = (255,255,255)
    black = (0,0,0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, position, font, .5, white, 1, cv2.LINE_AA)

def framesToTimestamp(frames):
    seconds = frames / 30
    minutes = round(seconds / 60)
    seconds = round(seconds % 60)
    return str(minutes) + ":" + str(seconds)

# sliding window for timing data
window_size = 100
window = []

# Last command 
last_command = "none"
last_command_sent = "none"
last_command_time = "0"

# Hold gesture for gesture_buffer frames before changing the command
gesture_buffer = 10
new_command = "none"
new_command_count = 0

success, img = cap.read()
frames = 1
print(img.shape)

while success:
    start_time = time.time()
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.emplaceAndPop([datum])
    img = datum.cvOutputData
    img = np.array(img, dtype=np.uint8)
    command = keypointsToCommand(datum.poseKeypoints)
    pos = keypointsToPosition(datum.poseKeypoints)
    teleop = keypointsToTeleop(datum.poseKeypoints)

    if "teleop" in command:
        command = teleop
    
    is_other_command = command != "none" and not "teleop" in command
    is_none_command = command == "none"
    is_teleop_command = "teleop" in command
    
    #print(command, frames)
    #print(command, "is teleop", is_teleop_command, "is none", is_none_command,
    #"is other", is_other_command)

    # We only want to send teleop or none command if we saw it 10 frames in a row
    use_command = False
    # We're evaluating if we should use this new command we've been tracking
    if not is_other_command and new_command == command:
        #print("New command and command are the same", command, new_command)
        new_command_count += 1
        # We saw it enough times, use it
        if new_command_count > gesture_buffer:
            use_command = True

    # We are seeing a new command        
    if command != last_command:
        # For non teleop commands, we want to transition from none
        if is_other_command and last_command_sent == "none":
            use_command = True

        # Smooth out none commands or teleop commands 
        new_command = command
        #print("Trying a new command", new_command.ljust(12), "at frame", frames)
        new_command_count = 1

    # We want to use this command        
    if use_command and command != last_command_sent:
        #print("sent command", command.ljust(12), "last command",
        #        last_command.ljust(12), "new command", new_command, "frame", frames)
        new_command = command
        new_command_count = 0
        # Communicate to server
        if use_server:
            print("sending command", command)
            ws.send(command)
        
        timestamp = framesToTimestamp(frames)
        last_command_sent = command
        last_command_time = timestamp
        print("sent command", command.ljust(12), "timestamp", timestamp)

    last_command = command

    end_time = time.time()

    window.append(end_time - start_time)
    if len(window) > window_size:
        window.pop(0)
    sec = sum(window) / len(window)
    ms = round(sec * 1000, 3)
    fps = round(1 / sec, 3)

    #print(ms, "ms,", fps, "fps", end="\r")
    #print(command, end="\r")

    if display or video:
        displayText(img, str(fps) + " FPS", (20,20))
        displayText(img, str(ms) + " ms per frame", (20,40))
        displayText(img, "Heuristics: " + str(command), (20, 60))
        displayText(img, "Model: " + str(teleop), (20, 80))
        displayText(img, "Position: " + str(pos), (20, 100))
        displayText(img, "Last sent: " + last_command_sent + " " +
                last_command_time, (20, 120))
        
        if video:
            out.write(img)
        else:    
            cv2.imshow('image',img)
            cv2.waitKey(1)

    # Prepare next frame
    success, img = cap.read()
    frames += 1

