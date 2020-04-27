#!/usr/bin/python3

import argparse
import sys
import cv2
import numpy as np
import pickle
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time

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
    return [x, y], output, frame


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


# find the back and front centers of the user from the given openpose keypoints
def find_user_centers(keypoints):
    # right heel, left heel
    back_keypoints = [11, 14]
    # left toes, right toes
    front_keypoints = [19, 22]
    back_coords = [0, 0]
    for keypoint in back_keypoints:
        back_coords[0] += keypoints[0][keypoint][0]
        back_coords[1] += keypoints[0][keypoint][1]
    front_coords = [0, 0]
    for keypoint in front_keypoints:
        front_coords[0] += keypoints[0][keypoint][0]
        front_coords[1] += keypoints[0][keypoint][1]
    # find center of back and front
    for i in range(len(back_coords)):
        back_coords[i] /= len(back_keypoints)
        front_coords[i] /= len(front_keypoints)
    return list(map(int, back_coords)), list(map(int, front_coords))


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
        back, front = find_user_centers(keypoints)
        back_x, back_y = check_coords(coords=back)
        front_x, front_y = check_coords(coords=front)
        if back_x and back_y:
            output = "User in frame at (" + str(y) + ", " + str(x) + ")"
            cv2.circle(frame, (back_x, back_y), 5, (0, 0, 255), -1)
        if front_x and front_y:
            cv2.circle(frame, (front_x, front_y), 5, (0, 0, 0), -1)
    return [back_x, back_y], [front_x, front_y], output, frame


# use openpose to get keypoint positions, also returning the modified frame
def get_keypoints(opWrapper, frame):
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    return datum.poseKeypoints, datum.cvOutputData


# calculates map coordinates from mapping
# returns position coordinates as a tuple and formatted string
# if x or y is none, returns none for coords and an empty string
def calc_coords(mapping, coords):
    x = coords[0]
    y = coords[1]
    if x is None or y is None:
        return None, ""
    frame_coord = np.float32([[[x, y]]])
    map_coord = cv2.perspectiveTransform(frame_coord, mapping).ravel()
    output = "On map at (" + str(int(map_coord[0])) + ", " + str(int(map_coord[1])) + ")"
    return (map_coord[0].item(), map_coord[1].item()), output


# converts a vector to cartesian coordinates given its slope and quadrant
def convert_vector_coord(slope, quadrant, distance):
    hypotenuse = np.sqrt(1 + slope ** 2)
    triangle_ratio = distance / hypotenuse
    x = quadrant * triangle_ratio
    y = quadrant * triangle_ratio * slope
    return [x, y]


# calculates point coords relative to user map coords
def calc_point_coords(back_coords, front_coords, grid_size, point_x, point_y):
    # meters to millimeters
    grid_size = 1000 * grid_size
    x_dist = point_x * grid_size
    y_dist = point_y * grid_size
    direction = [front_coords[i] - back_coords[i] for i in range(2)]
    # division by zero with floats will basically never happen
    if (direction[0] == 0 or direction[1] == 0):
        return None
    slope = direction[1] / direction[0]
    perpendicular_slope = -direction[0] / direction[1]
    y_delta = convert_vector_coord(slope, np.sign(direction[0]), y_dist)
    x_delta = convert_vector_coord(perpendicular_slope, np.sign(direction[1]), x_dist)
    point_coords = [sum(coord) for coord in zip(x_delta, y_delta, back_coords)]
    output = "Pointing at (" + str(point_x) + ", " + str(point_y) + ")"
    return point_coords, output



# writes user and robot coords onto the given frame
def write_coords(frame, user_frame_string, user_map_string, robot_frame_string, robot_map_string, point_string):
    color = (0, 0, 0)
    cv2.putText(frame, user_frame_string, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, user_map_string, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, point_string, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, robot_frame_string, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, robot_map_string, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
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


# initializes map with axes
def init_map(map_axes):
    fig, ax = plt.subplots()
    ax.axis("equal")
    fig.suptitle("Map")
    # plot axis lines
    x_axis = list(zip(map_axes[0][0], map_axes[0][1]))
    y_axis = list(zip(map_axes[1][0], map_axes[1][1]))
    plt.plot(x_axis[0], x_axis[1], color="#00ff00")
    plt.plot(y_axis[0], y_axis[1], color="#0000ff")


# draws the user and robot positions on the map and appends to frame
# removes the position circles from the map after append
def draw_map(frame, map_template, user_map_back_coords, user_map_front_coords, robot_map_coords, point_coords):
    fig = plt.gcf()
    ax = plt.gca()
    # plot user and robot positions if given
    if user_map_back_coords:
        user_back = plt.Circle(user_map_back_coords, 50, color="#0000ff")
        ax.add_patch(user_back)
    if user_map_front_coords:
        user_front = plt.Circle(user_map_front_coords, 50, color="#000000")
        ax.add_patch(user_front)
    if robot_map_coords:
        robot = plt.Circle(robot_map_coords, 50, color="#00ff00")
        ax.add_patch(robot)
    if point_coords:
        point = plt.Circle(point_coords, 50, color="#ff0000")
        ax.add_patch(point)
    fig.canvas.draw()
    # write plot to image and add to frame
    map_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    map_image = map_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if map_template is not None:
        map_image = cv2.addWeighted(map_image, 0.7, map_template, 0.3, 0)
    frame = np.concatenate((frame, map_image), axis=1)
    # restore map to original
    if user_map_back_coords:
        user_back.remove()
    if user_map_front_coords:
        user_front.remove()
    if robot_map_coords:
        robot.remove()
    if point_coords:
        point.remove()
    return frame


# scan video for user and robot and output pixel location per frame
def scan_video(video, opWrapper, mapping, map_file, sample_rate, points_file, video_output, display, fast):
    print("Scanning Video")
    positions = []
    points = None
    if points_file:
        with open(points_file, "r") as f:
            points = json.load(f)
    capture = cv2.VideoCapture(video)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if video_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_output, fourcc, fps / sample_rate, (2 * int(capture.get(3)), int(capture.get(4))))

    # set up map display
    map_axes, frame_axes = gen_axes(mapping)
    init_map(map_axes)
    map_template = None
    if map_file:
        map_template = cv2.imread(map_file)

    points_index = 0
    times = []
    # frame_index is index + 1
    for frame_index in tqdm(range(1, num_frames + 1)):
        start_time = time.time()
        success, frame = capture.read()
        # shouldnt happen
        if not success:
            break
        # make openpose aware of user movement per frame
        keypoints = None
        if not fast:
            keypoints, _ = get_keypoints(opWrapper, frame)
        # grab point data
        point_data = False
        point_string = "Not pointing"
        if points and points_index < len(points["frames"]) and points["frames"][points_index]["frame_number"] == frame_index:
            point_data = True
            point_x = points["frames"][points_index]["x"]
            point_y = points["frames"][points_index]["y"]
            points_index += 1
        # grab frames at sample rate
        if frame_index % sample_rate == 0:
            # find image coords
            user_back_coords, user_front_coords, user_frame_string, frame = find_user(opWrapper, frame, keypoints)
            robot_coords, robot_frame_string, frame = find_robot(frame)
            # find map coords
            user_map_back_coords, user_map_string = calc_coords(mapping, user_back_coords)
            user_map_front_coords, _ = calc_coords(mapping, user_front_coords)
            robot_map_coords, robot_map_string = calc_coords(mapping, robot_coords)
            # find point coords
            point_coords = None
            if point_data:
                point_coords, point_string = calc_point_coords(user_map_back_coords, user_map_front_coords, points["grid_size"], point_x, point_y)
            # update frame
            frame = write_coords(frame, user_frame_string, user_map_string, robot_frame_string, robot_map_string, point_string)
            frame = draw_axes(frame, frame_axes)
            frame = draw_map(frame, map_template, user_map_back_coords, user_map_front_coords, robot_map_coords, point_coords)
            # add to json
            position = {}
            position["frame_number"] = frame_index
            position["user_coords"] = user_map_back_coords
            position["robot_coords"] = robot_map_coords
            if point_data:
                position["point_coords"] = point_coords
            positions.append(position)
            if video_output:
                writer.write(frame)
            if display:
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)
        if frame_index > 100:
            times.append(time.time() - start_time)
    print("")
    print(np.median(times))
    print(np.max(times))
    return positions


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
            [x, y], _, _ = find_robot(frame)
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
        drift = []
        for i in range(len(image_coords)):
            coord = image_coords[i]
            np_coord = np.float32([[[coord[0], coord[1]]]])
            test_coord = cv2.perspectiveTransform(np_coord, mapping).ravel()
            test_coords.append(test_coord)
            diff = test_coord - map_coords[i]
            drift.append(np.sqrt(diff[0] ** 2 + diff[1] ** 2))
        test_coords = np.asarray(test_coords)

        # plot and save comparison
        plt.subplot(2, 2, 1)
        plt.title("Image Coords")
        plt.scatter(image_coords[:, 0], image_coords[:, 1])
        plt.subplot(2, 2, 2)
        plt.title("Map Coords")
        plt.scatter(map_coords[:, 0], map_coords[:, 1])
        plt.subplot(2, 2, 3)
        plt.title("Estimated Map Coords")
        plt.scatter(test_coords[:, 0], test_coords[:, 1])
        plt.subplot(2, 2, 4)
        plt.title("Mapping Drift")
        plt.plot(drift)
        plt.savefig(mapping_figure)
        print("Average Drift: ", round(np.median(drift), 2), "mm", sep="")
    return mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", help="video to run on")
    parser.add_argument("-c", "--coords", help="robot coordinates on map")
    parser.add_argument("-s", "--sample_rate", type=int, default=30, help="sample rate dictating frame skips")
    parser.add_argument("-o", "--output", help="file to store positions json")
    parser.add_argument("-vo", "--video_output", help="file to store modified video")
    parser.add_argument("-m", "--mapping", help="file to load pickled mapping from")
    parser.add_argument("-mo", "--mapping_output", help="file to store pickled mapping to")
    parser.add_argument("-mf", "--mapping_figure", help="file to store mapping estimate figure")
    parser.add_argument("-mt", "--map_template", help="map template to draw on")
    parser.add_argument("-d", "--display", action="store_true", default=False, help="display frames as they are processed")
    parser.add_argument("-f", "--fast", action="store_true", default=False, help="speed up openpose at the cost of accuracy")
    parser.add_argument("-p", "--points", help="file to load points json")

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
    positions = scan_video(args.video, opWrapper, mapping, args.map_template, args.sample_rate, args.points, args.video_output, args.display, args.fast)

    # write json file
    if args.output:
        with open(args.output, "w") as outfile:
            json.dump(positions, outfile)
