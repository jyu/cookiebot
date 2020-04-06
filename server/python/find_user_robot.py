#!/usr/bin/python3

import argparse
import sys
import cv2
import numpy as np
import pickle
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    lower = np.array([60, 70, 150], dtype="uint8")
    upper = np.array([170, 140, 255], dtype="uint8")
    # find the parts of the frame that matches the color range
    mask = cv2.inRange(frame, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    if len(contours) != 0:
        robot = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(robot)
        x = int(x)
        y = int(y)
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        output = "Robot in frame at (" + str(int(x)) + ", " + str(int(y)) + ")"
    return x, y, output, frame


# coords accepts a tuple and takes precedence over x and y
# return coords if coords are nonnegative and not None
# otherwise, return None for both coords
def check_coords(x=None, y=None, coords=None):
    if coords:
        x = coords[0]
        y = coords[1]
    if x and y:
        if x >= 0 and y >= 0:
            return x, y
    return None, None


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
        center = find_user_center(keypoints)
        x, y = check_coords(coords=center)
        if x and y:
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
# returns position coordinates as a tuple and formatted string
# if x or y is none, returns none for coords and an empty string
def calc_coords(mapping, x, y):
    if x is None or y is None:
        return None, ""
    frame_coord = np.float32([[[x, y]]])
    map_coord = cv2.perspectiveTransform(frame_coord, mapping).ravel()
    output = "On map at (" + str(int(map_coord[0])) + ", " + str(int(map_coord[1])) + ")"
    return (map_coord[0], map_coord[1]), output


# writes user and robot coords onto the given frame
def write_coords(frame, user_frame_string, user_map_string, robot_frame_string, robot_map_string):
    color = (0, 0, 0)
    cv2.putText(frame, user_frame_string, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, user_map_string, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, robot_frame_string, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, robot_map_string, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    return frame


# generates map and frame axes
def gen_axes(mapping):
    # create mapping from map coords to image coords
    inverse_mapping = np.linalg.inv(mapping)
    # select axes coords on map
    x_left_map = np.float32([[[-700, 0]]])
    x_right_map = np.float32([[[500, 0]]])
    y_bot_map = np.float32([[[0, -2000]]])
    y_top_map = np.float32([[[0, 100]]])
    # map axes coords to image
    x_left_frame = cv2.perspectiveTransform(x_left_map, inverse_mapping).ravel()
    x_right_frame = cv2.perspectiveTransform(x_right_map, inverse_mapping).ravel()
    y_bot_frame = cv2.perspectiveTransform(y_bot_map, inverse_mapping).ravel()
    y_top_frame = cv2.perspectiveTransform(y_top_map, inverse_mapping).ravel()
    x_left_frame = tuple(i for i in x_left_frame)
    x_right_frame = tuple(i for i in x_right_frame)
    y_bot_frame = tuple(i for i in y_bot_frame)
    y_top_frame = tuple(i for i in y_top_frame)
    map_axes = [[x_left_map.ravel(), x_right_map.ravel()], [y_bot_map.ravel(), y_top_map.ravel()]]
    frame_axes = [[x_left_frame, x_right_frame], [y_bot_frame, y_top_frame]]
    return map_axes, frame_axes


# draw frame axes on frame
def draw_axes(frame, frame_axes):
    # draw axes on frame
    cv2.line(frame, tuple(frame_axes[0][0]), tuple(frame_axes[0][1]), (0, 255, 0))
    cv2.line(frame, tuple(frame_axes[1][0]), tuple(frame_axes[1][1]), (0, 0, 255))
    return frame


# draws the map with axes and user and robot positions
def draw_map(frame, axes, user_map_coords, robot_map_coords):
    fig, ax = plt.subplots()
    ax.axis("equal")
    fig.suptitle("Map")
    # plot axis lines
    x_axis = list(zip(axes[0][0], axes[0][1]))
    y_axis = list(zip(axes[1][0], axes[1][1]))
    plt.plot(x_axis[0], x_axis[1], color="#00ff00")
    plt.plot(y_axis[0], y_axis[1], color="#0000ff")
    # plot user and robot positions if given
    if user_map_coords:
        user = plt.Circle(user_map_coords, 50, color="#0000ff")
        ax.add_patch(user)
    if robot_map_coords:
        robot = plt.Circle(robot_map_coords, 50, color="#00ff00")
        ax.add_patch(robot)
    fig.canvas.draw()
    # write plot to image and add to frame
    map_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    map_image = map_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frame = np.concatenate((frame, map_image), axis=1)
    plt.close(fig)
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
        writer = cv2.VideoWriter(video_output, fourcc, fps / sample_rate, (2 * int(capture.get(3)), int(capture.get(4))))

    map_axes, frame_axes = gen_axes(mapping)

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
            # find image coords
            user_x, user_y, user_frame_string, frame = find_user(opWrapper, frame, keypoints)
            robot_x, robot_y, robot_frame_string, frame = find_robot(frame)
            # find map coords
            user_map_coords, user_map_string = calc_coords(mapping, user_x, user_y)
            robot_map_coords, robot_map_string = calc_coords(mapping, robot_x, robot_y)
            # update frame
            frame = write_coords(frame, user_frame_string, user_map_string, robot_frame_string, robot_map_string)
            frame = draw_axes(frame, frame_axes)
            frame = draw_map(frame, map_axes, user_map_coords, robot_map_coords)
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
def build_mapping(video, coords_file, mapping_figure):
    print("Building Mapping")
    capture = cv2.VideoCapture(video)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    map_coords = np.loadtxt(coords_file)
    # select frames for mapping use
    frame_indices = np.linspace(0, num_frames - 1, len(map_coords), dtype=int)

    frame_coords = []
    next_frame = 0
    # select frames to be used for mapping using frame_indices
    for frame_index in tqdm(range(num_frames)):
        success, frame = capture.read()
        # shouldnt happen
        if not success:
            break
        # check if this frame is selected
        if frame_index == frame_indices[next_frame]:
            x, y, _, _ = find_robot(frame)
            x, y = check_coords(x, y)
            if x and y:
                frame_coords.append([x, y])
            # if this frame doesnt have the robot, delete the corresponding map coord
            else:
                map_coords = np.delete(map_coords, next_frame, 0)
            next_frame += 1
    image_coords = np.asarray(frame_coords)

    # create mapping from image plane to 2d map plane
    mapping, _ = cv2.findHomography(image_coords, map_coords)

    # test mapping on image coords and store results
    if mapping_figure:
        test_coords = []
        for coord in image_coords:
            np_coord = np.float32([[[coord[0], coord[1]]]])
            test_coords.append(cv2.perspectiveTransform(np_coord, mapping).ravel())
        test_coords = np.asarray(test_coords)

        # plot and save comparison
        plt.subplot(1, 3, 1)
        plt.title("Image Coords")
        plt.scatter(image_coords[:, 0], image_coords[:, 1])
        plt.subplot(1, 3, 2)
        plt.title("Map Coords")
        plt.scatter(map_coords[:, 0], map_coords[:, 1])
        plt.subplot(1, 3, 3)
        plt.title("Estimated Map Coords")
        plt.scatter(test_coords[:, 0], test_coords[:, 1])
        plt.savefig(mapping_figure)

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
    parser.add_argument("-mf", "--mapping_figure", help="file to store mapping estimate figure")
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
        mapping = build_mapping(args.video, args.coords, args.mapping_figure)

    # store mapping
    if args.mapping_output:
        with open(args.mapping_output, "wb") as mapping_file:
            pickle.dump(mapping, mapping_file)

    # scan video
    scan_video(args.video, opWrapper, mapping, args.sample_rate, args.video_output, args.output, args.display, args.fast)
