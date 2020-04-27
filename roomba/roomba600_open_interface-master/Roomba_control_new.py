import serial
import time
import keyboard
import threading
import math
import matplotlib.pyplot as plt
import numpy as np
import asyncio
import websockets
from multiprocessing import Process
from matplotlib import colors
# import webcam_script
import cv2
import base64
import asyncio
import websockets
import sys

update_interval = 1

class Robot_stat:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.dist = 0
        self.ang = 0
        self.ser = serial.Serial("COM5", baudrate=115200, timeout=1.0, writeTimeout=0)
        # self.ser = serial.Serial("COM5", baudrate=115200, timeout=0.5, writeTimeout=0)
        # self.ser = serial.Serial("COM5", baudrate=19200, timeout=0.5, writeTimeout=0)
        # self.ser = serial.Serial("/dev/ttyUSB0", baudrate=115200, timeout=0.5, writeTimeout=0)
        self.state = ""
        self.x_arr = np.array([0.0])
        self.y_arr = np.array([0.0])
        self.lightBumper = ""
        self.physBumper = ""
        self.home = (0.0,0.0)
        self.map = np.full((1,1),0)
        self.x_min = 0
        self.y_min = 0
        self.cell_size = 0
    def update(self, x, y, dist, ang, lightBumper, physBumper):
        self.x = x
        self.y = y
        # print("x:",x," y:",y)
        self.dist = dist
        self.ang = ang
        if (self.ang < 0):
            # self.ang = (2*math.pi) - self.ang
            self.ang = (2*math.pi) + self.ang
        if (self.ang > 2*math.pi):
            self.ang = self.ang - (2*math.pi)
        if (update_interval % 5 == 0):
            self.x_arr = np.append(self.x_arr,[x])
            self.y_arr = np.append(self.y_arr,[y])
            # self.x = x
            # self.y = y
        self.lightBumper = (bin(lightBumper)[2:]).zfill(6)
        self.physBumper = (bin(physBumper)[2:]).zfill(4)
    def print_stat(self):
        print("dist:", self.dist, " ang:", self.ang, " x:", self.x, " y:", self.y)
        # print("bump:", self.lightBumper)

stat = Robot_stat()

def odometry_fn():
    global stat
    global update_interval
    ser = stat.ser
    dist = 0
    ang = 0
    x = 0
    y = 0
    prev_encL = None
    prev_encR = None
    while True:
        # print("fuck you")
        # print("",end="")
        # print("x:",stat.x," y:",stat.y)
        # time.sleep(0.01)
        update_interval += 1
        # ser.write(bytes([150, 1]))
        data = None
        pdata1 = pdata2 = pdata3 = pdata4 = 0
        # print(ser.in_waiting)
        found = False

        while (found == False):
            # print("wtf")
            data = ser.read(1)
            data = int.from_bytes(data, byteorder='big', signed=False)
            if data == 19: # Header for data packet
                data = ser.read(1)
                data = int.from_bytes(data, byteorder='big', signed=False)
                if data == 10: # N-byte between header and checksum
                # if data == 2:
                    found = True
                    # ser.write(bytes([150, 0]))
                    # print("WHY WHY WHY")
                    # print(" ", end=" ")
                    # print(update_interval)
        if found:
            ser.read(1)
            pdata1 = ser.read(2)
            encoderL = int.from_bytes(pdata1, byteorder='big', signed=True)
            # encoderL = 0
            ser.read(1)
            pdata2 = ser.read(2)
            encoderR = int.from_bytes(pdata2, byteorder='big', signed=True)
            # encoderR = 0
            ser.read(1)
            pdata3 = ser.read(1)
            lightBumper = int.from_bytes(pdata3, byteorder='big', signed=False)
            ser.read(1)
            pdata4 = ser.read(1)
            physBumper = int.from_bytes(pdata4, byteorder='big', signed=False)
            # physBumper = 0
            # IDK
            encoderL = encoderL#*1.01
            encoderR = encoderR#*1.01
        """
        while ser.in_waiting:
            ser.read(1)
        ser.write(bytes([142, 43]))
        pdata1 = ser.read(2)
        encoderL = int.from_bytes(pdata1, byteorder='big', signed=True)
        ser.write(bytes([142, 44]))
        pdata2 = ser.read(2)
        encoderR = int.from_bytes(pdata2, byteorder='big', signed=True)
        ser.write(bytes([142, 45]))
        pdata3 = ser.read(1)
        lightBumper = int.from_bytes(pdata3, byteorder='big', signed=False)
        ser.write(bytes([142, 7]))
        pdata4 = ser.read(1)
        physBumper = int.from_bytes(pdata4, byteorder='big', signed=False)
        """

        if prev_encL is None:
            prev_encL = encoderL
            prev_encR = encoderR
        else:
            distL = (encoderL - prev_encL)*math.pi*72.0/508.8
            distR = (encoderR - prev_encR)*math.pi*72.0/508.8
            dist = (distL + distR)/2.0
            dist = dist#*0.9
            if abs(dist) > 1000:
                ser.write(bytes([137, 0, 0, 0, 0]))
                while True:
                    print("fucked up value received") 
                return
                # ser.reset_output_buffer()
                # ser.reset_input_buffer()
                # encoderL = prev_encL
                # encoderR = prev_encR
            else:
                # ang += ((distR - distL)/235.0)
                ang += ((distR - distL)/235.0)#*1.05

                if (ang < 0):
                    # self.ang = (2*math.pi) - self.ang
                    ang = (2*math.pi) + ang
                if (ang > 2*math.pi):
                    ang = ang - (2*math.pi)

                prev_encL = encoderL
                prev_encR = encoderR

                x -= dist*math.sin(ang)
                y += dist*math.cos(ang)

                stat.update(x, y, dist, ang, lightBumper, physBumper)
                print(lightBumper)
                # ser.write(bytes([150, 1]))
        # stat.print_stat()

def XY2Cell(x, y, x_min, y_min, cell_size):
    x_diff = x - x_min
    y_diff = y - y_min 
    cell_x = int(x_diff/cell_size)
    cell_y = int(y_diff/cell_size)
    if (cell_x <= 0):
        cell_x = 0
    if (cell_y <= 0):
        cell_y = 0
    return (cell_x, cell_y)

def createMap(cell_size):
    global stat
    x_min = np.amin(stat.x_arr)
    x_max = np.amax(stat.x_arr)
    y_min = np.amin(stat.y_arr)
    y_max = np.amax(stat.y_arr)
    x_cell_size = 1 + (XY2Cell(x_max, 0, x_min, y_min, cell_size))[0]
    y_cell_size = 1 + (XY2Cell(0, y_max, x_min, y_min, cell_size))[1]
    stat.map = np.full((y_cell_size, x_cell_size), 2)
    for i in range(np.size(stat.x_arr)):
        curr_x = stat.x_arr[i]
        curr_y = stat.y_arr[i]
        curr_xy_cell = (XY2Cell(curr_x, curr_y, x_min, y_min, cell_size))
        # left_xy_cell = (XY2Cell(curr_x - 174.25, curr_y, x_min, y_min, cell_size))
        # right_xy_cell = (XY2Cell(curr_x + 174.25, curr_y, x_min, y_min, cell_size))
        # front_xy_cell = (XY2Cell(curr_x, curr_y + 174.25, x_min, y_min, cell_size))
        # back_xy_cell = (XY2Cell(curr_x, curr_y - 174.25, x_min, y_min, cell_size))
        # if (stat.map[y_cell_size - 1 - curr_xy_cell[1], curr_xy_cell[0]] == 0):
        #     stat.map[y_cell_size - 1 - curr_xy_cell[1], curr_xy_cell[0]] = 1
        # elif (stat.map[y_cell_size - 1 - curr_xy_cell[1], curr_xy_cell[0]] == 1):
        #     stat.map[y_cell_size - 1 - curr_xy_cell[1], curr_xy_cell[0]] = 2
        # for i in range(left_xy_cell[0], right_xy_cell[0]):
        #     for j in range(back_xy_cell[1], front_xy_cell[1]):
        if (stat.map[curr_xy_cell[1], curr_xy_cell[0]] == 2):
            stat.map[curr_xy_cell[1], curr_xy_cell[0]] = 1
        elif (stat.map[curr_xy_cell[1], curr_xy_cell[0]] == 1):
            stat.map[curr_xy_cell[1], curr_xy_cell[0]] = 0
        # for i in range(x_cell_size):
        #     for j in range(y_cell_size):
        top_xy_cell = (XY2Cell(curr_x, curr_y + 184.25, x_min, y_min, cell_size))
        bottom_xy_cell = (XY2Cell(curr_x, curr_y - 184.25, x_min, y_min, cell_size))
        left_xy_cell = (XY2Cell(curr_x - 184.25, curr_y, x_min, y_min, cell_size))
        right_xy_cell = (XY2Cell(curr_x + 184.25, curr_y, x_min, y_min, cell_size))
        closest_top_xy_cell = -1
        closest_bot_xy_cell = -1
        closest_right_xy_cell = -1
        closest_left_xy_cell = -1

        for m in range(curr_xy_cell[1], top_xy_cell[1]):
            if m >= 0 and m < y_cell_size:
                if (stat.map[m, curr_xy_cell[0]] == 0):
                    closest_top_xy_cell = m
        for n in range(bottom_xy_cell[1], curr_xy_cell[1]):
            if n >= 0 and n < y_cell_size:
                if (stat.map[n, curr_xy_cell[0]] == 0):
                    closest_bot_xy_cell = n
        for p in range(left_xy_cell[0], curr_xy_cell[0]):
            if p >= 0 and p < x_cell_size:
                if (stat.map[curr_xy_cell[1], p] == 0):
                    closest_left_xy_cell = p
        for q in range(curr_xy_cell[0], right_xy_cell[0]):
            if q >= 0 and q < x_cell_size:
                if (stat.map[curr_xy_cell[1], q] == 0):
                    closest_right_xy_cell = q

        # if (stat.map[top_xy_cell[1], top_xy_cell[0]] == 0):
        if closest_top_xy_cell > 0:
            for k in range(curr_xy_cell[1], closest_top_xy_cell):
                stat.map[k, curr_xy_cell[0]] = 0
        # if (stat.map[bottom_xy_cell[1], bottom_xy_cell[0]] == 0):
        if closest_bot_xy_cell > 0:
            for k in range(bottom_xy_cell[1], curr_xy_cell[1]):
                stat.map[k, curr_xy_cell[0]] = 0
        if closest_left_xy_cell > 0:
            for k in range(closest_left_xy_cell, curr_xy_cell[0]):
                stat.map[curr_xy_cell[1], k] = 0
        if closest_right_xy_cell > 0:
            for k in range(curr_xy_cell[0], closest_right_xy_cell):
                stat.map[curr_xy_cell[1], k] = 0
    return


def isValid(x, y):
    global stat
    if stat.cell_size == 0:
        print("map not created")
        return False
    # x_min = np.amin(stat.x_arr)
    # y_min = np.amin(stat.y_arr)
    (x_cell, y_cell) = XY2Cell(x, y, stat.x_min, stat.y_min, stat.cell_size)
    if (stat.map[y_cell, x_cell] == 0):
        return True
    else:
        return False

def drive_fn():
    global stat
    ser = stat.ser
    while True:
        if (keyboard.is_pressed('up')):
            # ser.write(bytes([137, 0, 50, 0, 0]))
            ser.write(bytes([137, 0, 100, 128, 0]))
        elif (keyboard.is_pressed('down')):
            ser.write(bytes([137, 255, 156, 128, 0]))
        elif (keyboard.is_pressed('right')):
            ser.write(bytes([137, 0, 20, 255, 255]))
        elif (keyboard.is_pressed('left')):
            ser.write(bytes([137, 0, 20, 0, 1]))
        elif (keyboard.is_pressed('space')):
            ser.write(bytes([137, 0, 0, 0, 0]))
        elif (keyboard.is_pressed('d')):
            ser.write(bytes([143])) #dock
            # break

def wake_fn(stat):
    ser = stat.ser
    start_time = time.time()
    while True:
        elapsed = round(time.time() - start_time, 2)
        if elapsed > 50 and stat.mode == "passive":
            ser.write(bytes([128])) #Initiate passive mode
            ser.write(bytes([131])) #132:full 131:safe
            ser.write(bytes([128])) #return to passive mode
            elapsed = 0 
            start_time = time.time()

def mapping_fn():
    global stat
    ser = stat.ser
    print("map thread begun")
    start_x = None
    start_y = None
    # right = frontRight = centerRight = centerLeft = frontLeft = left = 0
    # leftBump = rightBump = 0
    steer_left = True
    left_origin = False
    done = False
    edge_done = False
    first_wall = False
    on_right = True
    epsilon = 0.08
    origin_err_goal = 150
    # ser.write(bytes([137, 0, 50, 0, 0])) # Steer left / Full speed
    # ser.write(bytes([137, 0, 100, 0, 0]))

    #For debug
    # start_x = stat.x
    # start_y = stat.y
    # edge_done = True
    # steer_left = False
    #For debug
    # time.sleep(5)

    charge_x = stat.x
    charge_y = stat.y
    ser.write(bytes([137, 255, 206, 128, 0]))
    time.sleep(5)
    ser.write(bytes([137, 0, 0, 0, 0]))
    stat.home = (stat.x + 70, stat.y)
    print(stat.home[0], " ", stat.home[1])

    ser.write(bytes([137, 0, 30, 255, 255]))
    # ser.write(bytes([137, 0, 30, 0, 1])) #For debug!

    time.sleep(3.5)
    ser.write(bytes([137, 0, 0, 0, 0]))
    start_x = stat.home[0]
    start_y = stat.home[1]
    has_begun = False

    #for debug
    # ser.write(bytes([137, 0, 30, 255, 255]))
    # time.sleep(3.5)
    # drive2(stat.home[0], stat.home[1])
    # steer_left = False # for debug
    # ser.write(bytes([137, 0, 20, 0, 1]))    
    # time.sleep(8.0) #12

    while done is False:
        # First edge following with steering left
        if steer_left is True and edge_done is False:
            if has_begun is False:
                ser.write(bytes([137, 0, 80, 4, 200]))
                has_begun = True
            if start_x != None:
                # err = abs(start_x - stat.x) + abs(start_y - stat.y)
                err = abs(charge_x - stat.x) + abs(stat.y - charge_y)
                print("err:", err)
                # time.sleep(0.2)
                if left_origin == False:
                    # if err > origin_err_goal:
                    if err > 1000:
                        left_origin = True
                if err < 400 and left_origin:
                    print("going to home")
                    drive2(stat.home[0], stat.home[1])
                    # while (err > origin_err_goal - 50):
                    #     time.sleep(0)
                    print("returned to origin")
                    left_origin = False
                    steer_left = False
                    has_begun = False
                    time.sleep(1.0)
                    ser.write(bytes([137, 0, 20, 0, 1]))    
                    time.sleep(3.0) #12
                    print("steering right now")
                    ser.write(bytes([137, 0, 0, 0, 0]))
                    # return
                    # edge_done = True # For debug

            if (int)(stat.lightBumper[3]) == 1 or (int)(stat.lightBumper[4]) == 1 or (int)(stat.lightBumper[5]) == 1:
                ser.write(bytes([137, 0, 20, 255, 255]))
                # print("what?")
                while (int)(stat.lightBumper[3]) == 1 or (int)(stat.lightBumper[4]) == 1 or (int)(stat.lightBumper[5]) == 1:
                    # print("turning")
                    time.sleep(0)
                # if start_x == None:
                #     start_x = stat.x
                #     start_y = stat.y
                ser.write(bytes([137, 0, 0, 0, 0])) # Is this necessary
                # time.sleep(0.5)
                ser.write(bytes([137, 0, 80, 4, 200]))
            else:
                # print("moving straight")
                # ser.write(bytes([137, 0, 100, 4, 200])) #2 200 -> 4 200 -> 4 0
                if int(stat.physBumper[3]) == 1:
                    # if start_x == None:
                    #     start_x = stat.x
                    #     start_y = stat.y
                    # print("right bumped")
                    ser.write(bytes([137, 255, 156, 0, 0]))
                    time.sleep(0.3)
                    ser.write(bytes([137, 0, 20, 255, 255]))
                    time.sleep(5.0)
                    ser.write(bytes([137, 0, 0, 0, 0]))
                    # time.sleep(0.5)
                    ser.write(bytes([137, 0, 80, 4, 200]))
                elif int(stat.physBumper[2]) == 1:
                    # if start_x == None:
                    #     start_x = stat.x
                    #     start_y = stat.y
                    # print("left bumped")
                    ser.write(bytes([137, 255, 156, 0, 0]))
                    time.sleep(0.3)
                    ser.write(bytes([137, 0, 20, 255, 255]))
                    time.sleep(0.4) #0.3
                    ser.write(bytes([137, 0, 0, 0, 0]))
                    # time.sleep(0.5)
                    ser.write(bytes([137, 0, 80, 4, 200]))
                # else:
                    # ser.write(bytes([137, 0, 100, 4, 200]))
            # print("steering left")
        elif steer_left is False and edge_done is False:
            # Second edge following with right steering
            # err = abs(start_x - stat.x) + abs(start_y - stat.y)
            err = abs(charge_x - stat.x) + abs(stat.y - charge_y)
            print("err:", err)
            if has_begun is False:
                ser.write(bytes([137, 0, 80, 252, 56]))
                has_begun = True
            if left_origin == False:
                # if err > origin_err_goal:
                if err > 1000:
                    left_origin = True
            # if err < origin_err_goal - 50 and left_origin:
            if err < 400 and left_origin:
                print("going to home")
                drive2(stat.home[0], stat.home[1])
                # while (err > origin_err_goal - 50):
                #     time.sleep(0)
                print("returned to origin")
                ser.write(bytes([137, 0, 0, 0, 0]))
                edge_done = True
                time.sleep(2)
                # start_x = stat.x
                # start_y = stat.y
                left_origin = False # Comment out this
                # break # and uncomment this for edge following only
            if (int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1 or (int)(stat.lightBumper[0]) == 1:
                ser.write(bytes([137, 0, 20, 0, 1]))
                while (int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1 or (int)(stat.lightBumper[0]) == 1:
                    print("turning")
                    # time.sleep(0)
                ser.write(bytes([137, 0, 0, 0, 0])) # Is this necessary
                ser.write(bytes([137, 0, 80, 252, 56]))
            else:
                # ser.write(bytes([137, 0, 100, 252, 56]))
                # ser.write(bytes([137, 0, 50, 252, 56]))
                if int(stat.physBumper[3]) == 1:
                    # print("right bumped")
                    ser.write(bytes([137, 255, 156, 0, 0]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 20, 0, 1]))
                    time.sleep(0.3) #0.3
                    # ser.write(bytes([137, 0, 50, 0, 0]))
                    ser.write(bytes([137, 0, 0, 0, 0]))
                    ser.write(bytes([137, 0, 80, 252, 56]))
                elif int(stat.physBumper[2]) == 1:
                    # print("left bumped")
                    ser.write(bytes([137, 255, 156, 0, 0]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 20, 0, 1]))
                    time.sleep(5.0) #2.5
                    # ser.write(bytes([137, 0, 50, 0, 0]))
                    ser.write(bytes([137, 0, 0, 0, 0]))
                    ser.write(bytes([137, 0, 80, 252, 56]))
                # print("steering right")
                # else:
                    # ser.write(bytes([137, 0, 100, 252, 56]))
        else:
            # Scanning the rest of the room
            print("scanning the rest")
            stat.y_min = np.amin(stat.y_arr) 
            # stat.y_min = 1500 # for debug
            err = abs(start_x - stat.x) + abs(start_y - stat.y)
            print("err:", err)
            # if left_origin == False:
            #     if err > 650:
            #         left_origin = True
            ser.write(bytes([137, 0, 20, 0, 1]))
            # print("turning left")
            # print(stat.ang)
            if (stat.ang > math.pi):
                ser.write(bytes([137, 0, 20, 0, 1]))
                while (stat.ang > math.pi):
                    time.sleep(0)
            while (stat.ang < (math.pi/2)):
                # print(stat.ang)
                time.sleep(0)
            ser.write(bytes([137, 0, 0, 0, 0]))
            # time.sleep(5)
            # ser.write(bytes([137, 50, 0, 0, 0]))
            to_left = True
            to_right = False
            on_wall = False
            while (abs(stat.y - stat.y_min) > 150):
                # print("time to mow the lawn")
                # print("going straight")
                # Go straight to the wall
                ser.write(bytes([137, 0, 100, 0, 0]))
                while ((int)(stat.lightBumper) == 0):
                    time.sleep(0)
                on_wall = True
                # print("found the wall")
                # print("tr:", to_right, "tl:", to_left)

                if to_left is True:
                    to_left = False
                    to_right = True
                elif to_right is True:
                    to_left = True
                    to_right = False

                start_t = time.time()
                goal_y = stat.y - 150
                # while (time.time() < start_t + 10):
                while (stat.y > goal_y) and (abs(stat.y - stat.y_min) > 150):
                    # print("move it move it")
                    if to_left is True:
                        # print("on right")
                        if (int)(stat.lightBumper[3]) == 1 or (int)(stat.lightBumper[4]) == 1 or (int)(stat.lightBumper[5]) == 1:
                            ser.write(bytes([137, 0, 20, 255, 255]))
                            while (int)(stat.lightBumper[3]) == 1 or (int)(stat.lightBumper[4]) == 1 or (int)(stat.lightBumper[5]) == 1:
                                time.sleep(0)
                            ser.write(bytes([137, 0, 0, 0, 0]))
                        else:
                            ser.write(bytes([137, 0, 100, 4, 200]))
                            if int(stat.physBumper[3]) == 1:
                                ser.write(bytes([137, 255, 156, 0, 0]))
                                time.sleep(0.2)
                                ser.write(bytes([137, 0, 20, 255, 255]))
                                time.sleep(5.0)
                                ser.write(bytes([137, 0, 0, 0, 0]))
                            elif int(stat.physBumper[2]) == 1:
                                ser.write(bytes([137, 255, 156, 0, 0]))
                                time.sleep(0.2)
                                ser.write(bytes([137, 0, 20, 255, 255]))
                                time.sleep(0.3) 
                                ser.write(bytes([137, 0, 0, 0, 0]))
                    else:
                        # print("on left")
                        if (int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1 or (int)(stat.lightBumper[0]) == 1:
                            ser.write(bytes([137, 0, 20, 0, 1]))
                            while (int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1 or (int)(stat.lightBumper[0]) == 1:
                                time.sleep(0)
                            ser.write(bytes([137, 0, 0, 0, 0]))
                        else:
                            ser.write(bytes([137, 0, 100, 252, 56]))
                            if int(stat.physBumper[3]) == 1:
                                ser.write(bytes([137, 255, 156, 0, 0]))
                                time.sleep(0.2)
                                ser.write(bytes([137, 0, 20, 0, 1]))
                                time.sleep(0.3)
                                ser.write(bytes([137, 0, 0, 0, 0]))
                            elif int(stat.physBumper[2]) == 1:
                                ser.write(bytes([137, 255, 156, 0, 0]))
                                time.sleep(0.2)
                                ser.write(bytes([137, 0, 20, 0, 1]))
                                time.sleep(5.0)
                                ser.write(bytes([137, 0, 0, 0, 0]))
                # print("moved around a bit")

                if to_left is True:
                    # print("let's go to the left")
                    ser.write(bytes([137, 0, 20, 255, 255]))
                    while (stat.ang > (math.pi/2)):
                        time.sleep(0)
                    ser.write(bytes([137, 0, 0, 0, 0]))
                    # to_left = False
                    on_wall = False
                elif to_right is True:
                    # print("let's go to the right")
                    ser.write(bytes([137, 0, 20, 0, 1]))
                    while (stat.ang < (3*math.pi/2)):
                        time.sleep(0)
                    ser.write(bytes([137, 0, 0, 0, 0]))
                    # to_right = False
                    on_wall = False
                # ser.write(bytes([137, 0, 0, 0, 0]))
                # ser.write(bytes([137, 100, 0, 0, 0]))
            ser.write(bytes([137, 0, 0, 0, 0]))
            time.sleep(5)
            print("map complete")
            drive2(stat.home[0], stat.home[1])
            break
def drive2(x, y):
    global stat
    ser = stat.ser
    dx = -stat.x + x
    dy = -stat.y + y
    theta = math.atan2(dy, dx) + (3*math.pi/2)
    if theta < 0:
        theta += 2*math.pi
    if theta > 2*math.pi:
        theta -= 2*math.pi
    dt = theta - stat.ang
    print("dx:", dx, " dy:", dy, " curr:", stat.ang, " goal theta:", theta, " dt:",dt)
    epsilon_t = 0.006
    epsilon_d = 20
    if (abs(dt) > math.pi and dt > 0) or (abs(dt) < math.pi and dt < 0):
        print("turning right")
        ser.write(bytes([137, 0, 20, 255, 255]))
        while(abs(dt) > epsilon_t):
            # ser.write(bytes([137, 0, 20, 0, 1]))
            dt = theta - stat.ang
            # print("err:", theta - stat.ang)
            # print("dx:", dx, " dy:", dy, " curr:", stat.ang, " goal theta:", theta, " dt:",dt)
            time.sleep(0)
    else:
        print("turning left")
        ser.write(bytes([137, 0, 20, 0, 1]))
        while(abs(dt) > epsilon_t):
            dt = theta - stat.ang
            # print("dx:", dx, " dy:", dy, " curr:", stat.ang, " goal theta:", theta, " dt:",dt)
            # ser.write(bytes([137, 0, 20, 255, 255]))
            time.sleep(0)
    print("done turning")
    # ser.write(bytes([137, 0, 0, 0, 0]))
    ser.write(bytes([137, 0, 50, 0, 0]))
    if x > stat.x:
        while x - stat.x > 0:
            time.sleep(0)
    else:
        while stat.x - x > 0:
            time.sleep(0)
    # while(abs(x - stat.x) + abs(y - stat.y) > epsilon_d):
    #     print(abs(x - stat.x) + abs(y - stat.y))
    #     time.sleep(0)
    print("turning")
    dt = - stat.ang
    if (abs(dt) > math.pi and dt > 0) or (abs(dt) < math.pi and dt < 0):
        print("turning right")
        ser.write(bytes([137, 0, 20, 255, 255]))
        while(abs(dt) > epsilon_t):
            # ser.write(bytes([137, 0, 20, 0, 1]))
            dt = -stat.ang
            # print("err:", theta - stat.ang)
            # print("dx:", dx, " dy:", dy, " curr:", stat.ang, " goal theta:", theta, " dt:",dt)
            time.sleep(0)
    else:
        print("turning left")
        ser.write(bytes([137, 0, 20, 0, 1]))
        while(abs(dt) > epsilon_t):
            dt = -stat.ang
            # print("dx:", dx, " dy:", dy, " curr:", stat.ang, " goal theta:", theta, " dt:",dt)
            # ser.write(bytes([137, 0, 20, 255, 255]))
            time.sleep(0)
    print("at home")
    ser.write(bytes([137, 0, 0, 0, 0]))
    return

def main1():
    # global robot
    global stat
    # robot = Robot_stat()
    # stat = Robot_stat()
    t_odometry = threading.Thread(target=odometry_fn, args=(), daemon=True)
    t_drive = threading.Thread(target=drive_fn, args=(), daemon=True)
    t_map = threading.Thread(target=mapping_fn, args=(), daemon=True)
    # t_map = threading.Thread(target=mapping_fn, args=())
    # t_drive2home = threading.Thread(target=drive2, args=(stat.home[0], stat.home[1]), daemon=True)

    # t_wake = threading.Thread(target=wake_fn, args=(stat,), daemon=True)
    # t_drive2home = threading.Thread(target=drive2, args=(), daemon=True)
    ser = stat.ser
    # Send "Start" Opcode to start Open Interface, Roomba in Passive Mode
    ser.write(bytes([128]))
    # ser.write(bytes([7]))
    # time.sleep(10)
    # ser.write(bytes([128]))
    stat.state = "passive"
    # Send "Safe Mode" Opcode to enable Roomba to respond to commands
    ser.write(bytes([131])) #132:full 131:safe
    # ser.write(bytes([148, 4, 19, 20, 45, 7])) # Initiate streaming
    ser.write(bytes([148, 4, 43, 44, 45, 7])) # Initiate streaming <- this one
    # ser.write(bytes([148, 1, 45])) # Initiate streaming
    stat.state = "safe"
    t_odometry.start()
    t_drive.start()
    # t_map.start()
    # t_wake.start()
    while True:
        # print("x:",stat.x," y:",stat.y)
        # ser.write(bytes([128]))
        # ser.write(bytes([131]))
        # print(ser.in_waiting)
        # print(t_map.is_alive() is False)
        # print(update_interval)
        if keyboard.is_pressed("esc"):
            ser.write(bytes([137, 0, 0, 0, 0]))
            ser.write(bytes([150, 0]))
            ser.write(bytes([128])) #return to passive mode
            # while ser.in_waiting:
                # print(ser.in_waiting)
            plt.subplot(1,2,1)
            data = {'x_data': stat.x_arr, 'y_data': stat.y_arr}
            # plt.plot('x_data','y_data',data=data)
            plt.scatter('x_data','y_data',data=data, s=2)
            plt.axis('scaled')
            # print(data)
            print(stat.x, " ", stat.y, " ", stat.ang)
            
            createMap(50)
            cmap = colors.ListedColormap(['white','gray', 'black'])
            # y_size = ((np.shape(stat.map))[0]) / ((np.shape(stat.map))[1])
            # y_size = y_size*3
            # plt.figure(figsize=(3,y_size))
            plt.subplot(1,2,2)
            # plt.pcolor(stat.map[::-1],cmap=cmap,edgecolors='k', linewidths=0.5)
            plt.pcolor(stat.map,cmap=cmap,edgecolors='k', linewidths=0.5)
            plt.show()
            np.savetxt('x_arr.txt', stat.x_arr, delimiter=',')
            np.savetxt('y_arr.txt', stat.y_arr, delimiter=',')
            np.savetxt('map.txt', stat.map, fmt='%s')
            break
        elif keyboard.is_pressed("m") and (t_map.is_alive() is False):
            print("map thread button pressed")
            t_map.start()
        # elif keyboard.is_pressed("h"):
        #     ser.write(bytes([137, 255, 206, 128, 0]))
        #     time.sleep(3.5)
        #     ser.write(bytes([137, 0, 0, 0, 0]))
        #     stat.home = (stat.x, stat.y)
        #     print(stat.home[0], " ", stat.home[1])
        #     ser.write(bytes([137, 0, 30, 255, 255]))
        #     time.sleep(2.5)
        #     ser.write(bytes([137, 0, 0, 0, 0]))
        #     t_drive2home = threading.Thread(target=drive2, args=(stat.home[0], stat.home[1]), daemon=True)
        # elif keyboard.is_pressed("h"):
            # drive2(stat.home[0], stat.home[1])
            # t_drive2home = threading.Thread(target=drive2, args=(stat.home[0], stat.home[1]), daemon=True)
            # t_drive2home.start()
            # drive2(0, 100)
    # plt.show()
    return

# def main():
#     global stat
#     t_odometry = threading.Thread(target=odometry_fn, args=(), daemon=True)
#     t_drive = threading.Thread(target=drive_fn, args=(), daemon=True)
#     ser = stat.ser
#     # Send "Start" Opcode to start Open Interface, Roomba in Passive Mode
#     ser.write(bytes([128]))
#     ser.write(bytes([131])) #132:full 131:safe
#     # ser.write(bytes([7]))
#     ser.write(bytes([148, 4, 43, 44, 45, 7])) # Initiate streaming
#     # ser.write(bytes([150, 0]))
#     # t_odometry.start()
#     t_drive.start()
#     while True:
#         # print("wtf")
#         # ser.write(bytes([142, 21]))
#         # # returns 1, for single byte in input buffer for Packet Id 21
#         # ser.in_waiting
#         # # read input buffer
#         # tmp = ser.read(1)
#         # convert byte response to integer
#         # res = int.from_bytes(tmp, byteorder='big', signed=False)
#         # will return 2 for Full Charging State
#         # print(res)
#         if keyboard.is_pressed("esc"):
#             ser.write(bytes([137, 0, 0, 0, 0]))
#             # ser.write(bytes([150, 0]))
#             ser.write(bytes([128])) #return to passive mode
#             return
#         time.sleep(0)

def main():
    # t_odometry = threading.Thread(target=odometry_fn, args=(), daemon=True)
    # t_drive = threading.Thread(target=drive_fn, args=(), daemon=True)
    # t_map = threading.Thread(target=mapping_fn, args=(), daemon=True)
    global stat
    global update_interval
    ser = stat.ser
    ser.write(bytes([128]))
    stat.state = "passive"
    ser.write(bytes([131])) #132:full 131:safe
    # ser.write(bytes([148, 1, 45])) # Initiate streaming
    ser.write(bytes([148, 4, 43, 44, 45, 7]))
    stat.state = "safe"
    prev_encL = None
    prev_encR = None
    ang = 0
    x = 0
    y = 0
    start_x = None
    start_y = None
    # right = frontRight = centerRight = centerLeft = frontLeft = left = 0
    # leftBump = rightBump = 0
    steer_left = True
    left_origin = False
    done = False
    edge_done = False
    first_wall = False
    on_right = True
    epsilon = 0.08
    origin_err_goal = 150
    # t_odometry.start()
    # t_drive.start()
    while True:
        # found = False
        # while (found == False):
        #     # print("wtf")
        #     data = ser.read(1)
        #     data = int.from_bytes(data, byteorder='big', signed=False)
        #     if data == 19: # Header for data packet
        #         data = ser.read(1)
        #         data = int.from_bytes(data, byteorder='big', signed=False)
        #         if data == 2: # N-byte between header and checksum
        #             found = True
        # if found:
        #     ser.read(1)
        #     pdata4 = ser.read(1)
        #     Bumper = int.from_bytes(pdata4, byteorder='big', signed=False)
        #     print(Bumper)
        if keyboard.is_pressed("esc"):
            ser.write(bytes([150, 0]))
            ser.write(bytes([128])) #return to passive mode
            plt.subplot(1,2,1)
            data = {'x_data': stat.x_arr, 'y_data': stat.y_arr}
            # plt.plot('x_data','y_data',data=data)
            plt.scatter('x_data','y_data',data=data, s=2)
            plt.axis('scaled')
            # print(data)
            print(stat.x, " ", stat.y, " ", stat.ang)
            
            createMap(50)
            cmap = colors.ListedColormap(['white','gray', 'black'])
            # y_size = ((np.shape(stat.map))[0]) / ((np.shape(stat.map))[1])
            # y_size = y_size*3
            # plt.figure(figsize=(3,y_size))
            plt.subplot(1,2,2)
            # plt.pcolor(stat.map[::-1],cmap=cmap,edgecolors='k', linewidths=0.5)
            plt.pcolor(stat.map,cmap=cmap,edgecolors='k', linewidths=0.5)
            plt.show()
            np.savetxt('x_arr.txt', stat.x_arr, delimiter=',')
            np.savetxt('y_arr.txt', stat.y_arr, delimiter=',')
            np.savetxt('map.txt', stat.map, fmt='%s')
            return
        if (keyboard.is_pressed('up')):
            # ser.write(bytes([137, 0, 50, 0, 0]))
            ser.write(bytes([137, 0, 100, 128, 0]))
        elif (keyboard.is_pressed('down')):
            ser.write(bytes([137, 255, 156, 128, 0]))
        elif (keyboard.is_pressed('right')):
            ser.write(bytes([137, 0, 20, 255, 255]))
        elif (keyboard.is_pressed('left')):
            ser.write(bytes([137, 0, 20, 0, 1]))
        elif (keyboard.is_pressed('space')):
            ser.write(bytes([137, 0, 0, 0, 0]))
        elif (keyboard.is_pressed('d')):
            ser.write(bytes([143])) #dock

        pdata1 = pdata2 = pdata3 = pdata4 = 0
        found = False
        while (found == False):
            data = ser.read(1)
            data = int.from_bytes(data, byteorder='big', signed=False)
            if data == 19: # Header for data packet
                data = ser.read(1)
                data = int.from_bytes(data, byteorder='big', signed=False)
                if data == 10: # N-byte between header and checksum
                    found = True
                    update_interval += 1
        if found:
            ser.read(1)
            pdata1 = ser.read(2)
            encoderL = int.from_bytes(pdata1, byteorder='big', signed=True)
            # encoderL = 0
            ser.read(1)
            pdata2 = ser.read(2)
            encoderR = int.from_bytes(pdata2, byteorder='big', signed=True)
            # encoderR = 0
            ser.read(1)
            pdata3 = ser.read(1)
            lightBumper = int.from_bytes(pdata3, byteorder='big', signed=False)
            ser.read(1)
            pdata4 = ser.read(1)
            physBumper = int.from_bytes(pdata4, byteorder='big', signed=False)
            # physBumper = 0
            # IDK
            encoderL = encoderL#*1.01
            encoderR = encoderR#*1.01

        if prev_encL is None:
            prev_encL = encoderL
            prev_encR = encoderR
        else:
            distL = (encoderL - prev_encL)*math.pi*72.0/508.8
            distR = (encoderR - prev_encR)*math.pi*72.0/508.8
            dist = (distL + distR)/2.0
            dist = dist#*0.9
            if abs(dist) > 1000:
                ser.write(bytes([137, 0, 0, 0, 0]))
                while True:
                    print("fucked up value received") 
                return
            else:
                # ang += ((distR - distL)/235.0)
                ang += ((distR - distL)/235.0)#*1.05

                if (ang < 0):
                    # self.ang = (2*math.pi) - self.ang
                    ang = (2*math.pi) + ang
                if (ang > 2*math.pi):
                    ang = ang - (2*math.pi)

                prev_encL = encoderL
                prev_encR = encoderR

                x -= dist*math.sin(ang)
                y += dist*math.cos(ang)

                stat.update(x, y, dist, ang, lightBumper, physBumper)
                print(stat.lightBumper)


if __name__ == "__main__":
    main()