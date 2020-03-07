import cv2
import subprocess
import sys
import numpy as np
import time
from websocket import create_connection
import argparse
import os

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
    
    # Teleop    
    teleop_mode = False
    delta_tolerance = 30

    # Teleop mode: when right elbow is raised, and wrist is not too high above
    # elbow
    if (abs(res["r_elbow"][y] - res["r_shoulder"][y]) < delta_tolerance and
        abs(res["r_wrist"][y] - res["r_elbow"][y]) < delta_tolerance * 2):
        return "teleop"

    return "none"

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action="store_true", help="show display")
    parser.add_argument('-teleop', action="store_true", help='recording for teleop, otherwise recording starts automatically')
    parser.add_argument('-mode', help='what gesture is being recorded')
    parser.add_argument('-dir', help='directory to save the features')

    args = parser.parse_args()
    display = args.d
    teleop_only = args.teleop
    mode = args.mode
    feat_dir = args.dir
    
    frames_until_capture = 200
    frames_to_capture = 300

    if mode == None:
        print("No mode for collecting data found, exiting..")
        exit()

    print("COLLECTING DATA FOR MODE", mode)

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
        white = (255, 255, 255) 
        black = (0, 0, 0)
        if display:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(fps) + " FPS", (20, 20), font, .5, white, 1, cv2.LINE_AA)
            cv2.putText(img, str(ms) + " ms per frame", (20, 50), font, .5, white, 1, cv2.LINE_AA)
            cv2.putText(img, str(command), (20, 100), font, .5, black, 1, cv2.LINE_AA)
            
            skip = False
            if not teleop_only:
                if frames_until_capture > 0:
                    until_msg = str(frames_until_capture) + " frames until capture"
                    cv2.putText(img, until_msg, (20, 150), font, .5, white, 1, cv2.LINE_AA)
                    skip = True
                    frames_until_capture -= 1
                else:
                    msg = str(written) + " frames captured"
                    cv2.putText(img, msg, (20, 150), font, .5, white, 1, cv2.LINE_AA)

            cv2.imshow('image',img)
            key = cv2.waitKey(1)
            if skip:
                continue

            # Auto method

            # Save as feat only on 5th frame
            if frames % 10 != 0:
                continue

            # Must follow teleop heuristic
            if teleop_only and not "teleop" in command:
                continue

            feat = getKeyPointsFeat(keypoints)

            if feat is None:
                continue

            print(feat)
            print(feat.shape)
            f = mode
            fwrite = open(feat_dir + "/" + f, 'a')
            line = str(feat[0])
            for m in range(1, feat.shape[0]):
                line += ";" + str(feat[m])
            line += "\n"
            fwrite.write(line)
            fwrite.close()
            written += 1
            print("written", written)
            
            # For all other methods, record a set number of frames
            if not teleop_only and written > frames_to_capture:
                print("Done!")
                exit()
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
                
