#!/usr/bin/python3

import argparse
import sys
import cv2
import numpy as np
import pickle
import math
from tqdm import tqdm

# find openpose library
openpose_path = "../../../openpose"
openpose_python_path = openpose_path + "/build/python"
sys.path.append(openpose_python_path)

from openpose import pyopenpose as op


# find the position of the robot in the given frame
# returns position coordinates, formatted string, and modified frame
# coordinates will be None if robot not found
def find_robot(frame):
    x = None
    y = None
    output = "Robot not found"
    # boundaries for the color of the robot in BGR
    lower = np.array([50, 50, 130], dtype="uint8")
    upper = np.array([175, 120, 225], dtype="uint8")
    # find the parts of the frame that matches the color range
    mask = cv2.inRange(frame, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        robot = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(robot)
        x = int(x)
        y = int(y)
        cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), -1)
        output = "Robot in frame at (" + str(int(x)) + ", " + str(int(y)) + ")"
    return x, y, output, frame


# find the center of the user from the given openpose keypoints
def find_user_center(keypoints):
    # right heel, left heel, left toes, right toes
    relevant_keypoints = [11, 14, 19, 22]
    x = 0
    y = 0
    for keypoint in relevant_keypoints:
        x += keypoints[0][keypoint][0]
        y += keypoints[0][keypoint][1]
    x /= len(relevant_keypoints)
    y /= len(relevant_keypoints)
    return int(x), int(y)


# find the position of the user in the given frame
# returns position coordinates, formatted string, and modified frame
# coordinates will be None if user not found
def find_user(opWrapper, frame, keypoints):
    x = None
    y = None
    output = "User not found"
    # find keypoints if not provided
    if keypoints is None:
        keypoints, _ = get_keypoints(opWrapper, frame)
    # check if the user was found
    if np.size(keypoints) != 1:
        x, y = find_user_center(keypoints)
        output = "User in frame at (" + str(y) + ", " + str(x) + ")"
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    return x, y, output, frame


# use openpose to get keypoint positions, also returning the modified frame
def get_keypoints(opWrapper, frame):
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    return datum.poseKeypoints, datum.cvOutputData


# calculates map coordinates from mapping
# returns position coordinates and formatted string
# if x or y is none, returns none for coords and an empty string
def calc_coords(mapping, x, y):
    if x is None or y is None:
        return None, None, ""
    frame_coord = np.array([[[x, y]]], dtype=np.float32)
    map_coord = cv2.perspectiveTransform(frame_coord, mapping).ravel()
    output = "User on map at (" + str(int(map_coord[0])) + ", " + str(int(map_coord[1])) + ")"
    return map_coord[0], map_coord[1], output


# writes user and robot coords onto the given frame
def write_coords(frame, user_frame_string, user_map_string, robot_frame_string, robot_map_string):
    cv2.putText(frame, user_frame_string, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, user_map_string, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, robot_frame_string, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, robot_map_string, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# scan video for user and robot and output pixel location per frame
def scan_video(video, opWrapper, mapping, sample_rate, video_output, output, display, fast):
    print("Scanning Video")
    outfile = None
    if output:
        outfile = open(output, "w")

    capture = cv2.VideoCapture(video)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if video_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_output, fourcc, fps / sample_rate, (int(capture.get(3)), int(capture.get(4))))

    # frame_index is index + 1
    for frame_index in tqdm(range(1, num_frames + 1)):
        success, frame = capture.read()
        # shouldnt happen
        if not success:
            break
        # make openpose aware of user movement per frame
        keypoints = None
        if not fast:
            keypoints, _ = get_keypoints(opWrapper, frame)
        # grab 1 frame a second
        if frame_index % sample_rate == 0:
            user_x, user_y, user_frame_string, frame = find_user(opWrapper, frame, keypoints)
            robot_x, robot_y, robot_frame_string, frame = find_robot(frame)
            _, _, user_map_string = calc_coords(mapping, user_x, user_y)
            _, _, robot_map_string = calc_coords(mapping, robot_x, robot_y)
            frame = write_coords(frame, user_frame_string, user_map_string, robot_frame_string, robot_map_string)
            if video_output:
                writer.write(frame)
            if output:
                outfile.write("Frame " + str(frame_index) + "\n")
                outfile.write(user_frame_string + "\n")
                outfile.write(user_map_string + "\n")
                outfile.write(robot_frame_string + "\n")
                outfile.write(robot_map_string + "\n")
                outfile.write("\n")
            if display:
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)
    print("")
    if output:
        outfile.close()


# builds a mapping between camera plane and map plane
# returns homography
def build_mapping(video, coords_file):
    print("Building Mapping")
    capture = cv2.VideoCapture(video)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    coords = np.loadtxt(coords_file)
    num_coords = len(coords)
    ratio = math.ceil(num_frames / num_coords)

    frame_coords = []
    # select num_coords frames to be used for mapping
    # frame_index is index + 1
    for frame_index in tqdm(range(1, num_frames + 1)):
        success, frame = capture.read()
        # shouldnt happen
        if not success:
            break
        # select a roughly even spread across all frames
        if int(frame_index / (len(frame_coords) + 1)) == ratio:
            x, y, _, _ = find_robot(frame)
            frame_coords.append([x, y])
    print("")
    # coordinates in the first plane (camera view)
    image_coords = np.asarray(frame_coords)
    # coordinates in the plane to map to (2d map)
    map_coords = coords[0:len(frame_coords)]

    mapping, _ = cv2.findHomography(image_coords, map_coords)
    return mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", help="video to run on")
    parser.add_argument("-c", "--coords", help="robot coordinates on map")
    parser.add_argument("-s", "--sample_rate", type=int, default=30, help="sample rate dictating frame skips")
    parser.add_argument("-o", "--output", help="file to store results")
    parser.add_argument("-vo", "--video_output", help="file to store modified video")
    parser.add_argument("-m", "--mapping", help="file to load pickled mapping from")
    parser.add_argument("-mo", "--mapping_output", help="file to store pickled mapping to")
    parser.add_argument("-d", "--display", action="store_true", default=False, help="display frames as they are processed")
    parser.add_argument("-f", "--fast", action="store_true", default=False, help="speed up openpose at the cost of accuracy")

    args = parser.parse_args()

    if args.video is None:
        print("Video not specified")
        sys.exit()

    # openpose params
    op_params = {}
    op_params["model_folder"] = openpose_path + "/models/"
    op_params["tracking"] = 5
    op_params["number_people_max"] = 1

    # start openpose
    opWrapper = op.WrapperPython()
    opWrapper.configure(op_params)
    opWrapper.start()
    print("")

    # get mapping
    mapping = None
    if args.mapping:
        print("Loading Mapping\n")
        with open(args.mapping, "rb") as mapping_file:
            mapping = pickle.load(mapping_file)
    else:
        if args.coords is None:
            print("Robot Coords not specified")
            sys.exit()
        mapping = build_mapping(args.video, args.coords)

    # store mapping
    if args.mapping_output:
        with open(args.mapping_output, "wb") as mapping_file:
            pickle.dump(mapping, mapping_file)

    # scan video
    scan_video(args.video, opWrapper, mapping, args.sample_rate, args.video_output, args.output, args.display, args.fast)
