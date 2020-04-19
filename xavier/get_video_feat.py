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

def getKeyPointsLocationFeat(keypoints, img_shape):

    if len(keypoints.shape) == 0:
        return None

    height, width, _ = img_shape

    person = keypoints[0]
    x = 0
    y = 1
    chest = person[1]

    feat = [chest[x]/width, chest[y]/height]
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
    parser.add_argument('-vdir', help='directory to for videos')
    parser.add_argument('-dir', help='directory to save the keypoint features')
    parser.add_argument('-ldir', help='directory to save the location features')

    args = parser.parse_args()
    feat_dir = args.dir
    video_dir = args.vdir
    loc_dir = args.ldir
    
    teleop_only = "teleop" in feat_dir

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

    # Log files for which features have been extracted from
    log = open('log.txt', 'r')
    completed = list(log.readlines())
    completed = list(map(lambda x: x.replace('\n', ''), completed))
    log.close()
    log = open('log.txt', 'a')
    
    # Expects directory of class directories of videos
    classes = os.listdir(video_dir)
    for c in classes:
        videos = os.listdir(video_dir + '/' + c)

        fwrite = open(feat_dir + "/" + c, 'a')
        fwrite_loc = open(loc_dir + "/" + c, 'a')

        for video in videos:
            path = video_dir + "/" + c + "/" + video
            if path in completed:
                continue
            print("Using video as input from path", path)
            log.write(path + '\n')

            cap = cv2.VideoCapture(path)
            out_f = "feat_out/" + c + '_' + video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_f, fourcc, 30.0/3, (int(cap.get(3)), int(cap.get(4))))

            frames = 0
            success = True
            written = 0
            
            while success:
                success, img = cap.read()
                if not success:
                    break
                frames += 1
                # Run on every frame to improve accuracy
                datum = op.Datum()
                datum.cvInputData = img
                opWrapper.emplaceAndPop([datum])
                img = datum.cvOutputData
                # Save as feat only on 3th frame
                if frames % 3 == 0:
                    img = np.array(img, dtype=np.uint8)
                    keypoints = datum.poseKeypoints
                    #command = keypointsToCommand(datum.poseKeypoints)

                    #white = (255, 255, 255) 

                    #font = cv2.FONT_HERSHEY_SIMPLEX
                    #cv2.putText(img, str(command), (20, 20), font, .5, white, 1, cv2.LINE_AA)
                    out.write(img)    

                    # For teleop only, ust follow teleop heuristic
                    #if teleop_only and not "teleop" in command:
                    #    continue

                    feat = getKeyPointsFeat(keypoints)
                    loc = getKeyPointsLocationFeat(keypoints, img.shape)

                    if feat is None:
                        continue

                    line = str(feat[0])
                    for m in range(1, feat.shape[0]):
                        line += ";" + str(feat[m])
                    line += "\n"
                    fwrite.write(line)

                    feat = loc
                    line = str(feat[0])
                    for m in range(1, feat.shape[0]):
                        line += ";" + str(feat[m])
                    line += "\n"
                    fwrite_loc.write(line)

                    written += 1
                    print("written", written, "frame", frames, end="\r")
        fwrite.close()
        fwrite_loc.close()
