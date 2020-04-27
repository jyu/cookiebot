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
        # self.ser = serial.Serial("COM5", baudrate=115200, timeout=55.5, writeTimeout=55.5)
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
        self.lightBumper = (bin(lightBumper)[2:]).zfill(6)
        self.physBumper = (bin(physBumper)[2:]).zfill(4)
    def print_stat(self):
        print("dist:", self.dist, " ang:", self.ang, " x:", self.x, " y:", self.y)
        # print("bump:", self.lightBumper)

stat = Robot_stat()

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
        if (stat.map[curr_xy_cell[1], curr_xy_cell[0]] == 2):
            stat.map[curr_xy_cell[1], curr_xy_cell[0]] = 1
        elif (stat.map[curr_xy_cell[1], curr_xy_cell[0]] == 1):
            stat.map[curr_xy_cell[1], curr_xy_cell[0]] = 0
        top_xy_cell = (XY2Cell(curr_x, curr_y + 154.25, x_min, y_min, cell_size))
        bottom_xy_cell = (XY2Cell(curr_x, curr_y - 154.25, x_min, y_min, cell_size))
        left_xy_cell = (XY2Cell(curr_x - 154.25, curr_y, x_min, y_min, cell_size))
        right_xy_cell = (XY2Cell(curr_x + 154.25, curr_y, x_min, y_min, cell_size)) #184.25
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

        if closest_top_xy_cell > 0:
            for k in range(curr_xy_cell[1], closest_top_xy_cell):
                stat.map[k, curr_xy_cell[0]] = 0
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
            dt = theta - stat.ang
            time.sleep(0)
    else:
        print("turning left")
        ser.write(bytes([137, 0, 20, 0, 1]))
        while(abs(dt) > epsilon_t):
            dt = theta - stat.ang
            time.sleep(0)
    print("done turning")
    ser.write(bytes([137, 0, 50, 0, 0]))
    if x > stat.x:
        while x - stat.x > 0:
            time.sleep(0)
    else:
        while stat.x - x > 0:
            time.sleep(0)
    print("turning")
    dt = - stat.ang
    if (abs(dt) > math.pi and dt > 0) or (abs(dt) < math.pi and dt < 0):
        print("turning right")
        ser.write(bytes([137, 0, 20, 255, 255]))
        while(abs(dt) > epsilon_t):
            dt = -stat.ang
            time.sleep(0)
    else:
        print("turning left")
        ser.write(bytes([137, 0, 20, 0, 1]))
        while(abs(dt) > epsilon_t):
            dt = -stat.ang
            time.sleep(0)
    print("at home")
    ser.write(bytes([137, 0, 0, 0, 0]))
    return

def main1():
    global stat
    global update_interval
    ser = stat.ser
    ser.write(bytes([128]))
    # ser.write(bytes([7]))
    # time.sleep(10)
    stat.state = "passive"
    ser.write(bytes([131])) #132:full 131:safe
    ser.write(bytes([148, 4, 43, 44, 45, 7]))
    stat.state = "safe"
    prev_encL = None
    prev_encR = None
    ang = 0
    x = 0
    y = 0
    start_x = None
    start_y = None
    steer_left = True
    left_origin = False
    done = False
    edge_done = False
    first_wall = False
    on_right = True
    epsilon = 0.08
    origin_err_goal = 150
    map_start = False
    initiated = False
    move_back = False
    epsilon_t = 0.006
    reached = False
    lawn_done = False
    while True:
        if keyboard.is_pressed("esc") or done is True:
            print("exiting")
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
            plt.subplot(1,2,2)
            plt.pcolor(stat.map,cmap=cmap,edgecolors='k', linewidths=0.5)
            plt.show()
            np.savetxt('x_arr.txt', stat.x_arr, delimiter=',')
            np.savetxt('y_arr.txt', stat.y_arr, delimiter=',')
            np.savetxt('map.txt', stat.map, fmt='%s')
            return
            
        if (keyboard.is_pressed('up')):
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
        elif (keyboard.is_pressed('m')):
            map_start = True

        pdata1 = pdata2 = pdata3 = pdata4 = 0
        found = False
        # while (update_interval % 5 != 0):
        while (found == False):
            # print("here?")
            data = stat.ser.read(1)
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
            ser.read(1)
            pdata2 = ser.read(2)
            encoderR = int.from_bytes(pdata2, byteorder='big', signed=True)
            ser.read(1)
            pdata3 = ser.read(1)
            lightBumper = int.from_bytes(pdata3, byteorder='big', signed=False)
            ser.read(1)
            pdata4 = ser.read(1)
            physBumper = int.from_bytes(pdata4, byteorder='big', signed=False)
            encoderL = encoderL
            encoderR = encoderR
        if prev_encL is None:
            prev_encL = encoderL
            prev_encR = encoderR
        else:
            distL = (encoderL - prev_encL)*math.pi*72.0/508.8
            distR = (encoderR - prev_encR)*math.pi*72.0/508.8
            dist = (distL + distR)/2.0
            dist = dist
            if abs(dist) > 1000:
                ser.write(bytes([137, 0, 0, 0, 0]))
                print("fucked up value received") 
                # return
                done = True
            else:
                ang += ((distR - distL)/235.0)
                if (ang < 0):
                    ang = (2*math.pi) + ang
                if (ang > 2*math.pi):
                    ang = ang - (2*math.pi)
                prev_encL = encoderL
                prev_encR = encoderR
                x -= dist*math.sin(ang)
                y += dist*math.cos(ang)
                stat.update(x, y, dist, ang, lightBumper, physBumper)
                # print(stat.lightBumper)
        if map_start is True and done is False:
            if initiated is False:
                charge_x = stat.x
                charge_y = stat.y
                # ser.write(bytes([137, 255, 206, 128, 0]))
                ser.write(bytes([137, 255, 206, 0, 0]))
                time.sleep(5)
                ser.write(bytes([137, 0, 0, 0, 0]))
                # stat.home = (stat.x + 70, stat.y)
                stat.home = (10, -285)
                print(stat.home[0], " ", stat.home[1])
                # ser.write(bytes([137, 0, 30, 255, 255]))
                # ser.write(bytes([137, 0, 30, 0, 1])) #For debug!
                # time.sleep(3.5)
                # ser.write(bytes([137, 0, 0, 0, 0]))
                start_x = stat.home[0]
                start_y = stat.home[1]
                has_begun = False
                initiated = True
                goal_calculated = False
                turned = False
                steer_left = False # for debug
                # edge_done = True #for debug
                lawn_done = False
                lawn_initiated = False
            else:
                if steer_left is True and edge_done is False:
                    if has_begun is False:
                        ser.write(bytes([137, 0, 80, 4, 200]))
                        has_begun = True
                    if start_x != None:
                        err = abs(charge_x - stat.x) + abs(stat.y - charge_y)
                        # print("err:", err)
                        if left_origin == False:
                            if err > 1000:
                                print("left origin")
                                left_origin = True
                        if err < 500 and left_origin:
                            print("going to home")
                            if goal_calculated is False:
                                dx = -stat.x + stat.home[0]
                                dy = -stat.y + stat.home[1]
                                theta = math.atan2(dy, dx) + (3*math.pi/2)
                                if theta < 0:
                                    theta += 2*math.pi
                                if theta > 2*math.pi:
                                    theta -= 2*math.pi
                                dt = theta - stat.ang
                                goal_calculated = True
                            elif turned is False:
                                dt = theta - stat.ang
                                if (abs(dt) < epsilon_t):
                                    turned = True
                                if (abs(dt) > math.pi and dt > 0) or (abs(dt) < math.pi and dt < 0):
                                    print("turning right")
                                    ser.write(bytes([137, 0, 20, 255, 255]))
                                else:
                                    print("turning left")
                                    ser.write(bytes([137, 0, 20, 0, 1]))
                            elif reached is False:
                                print("heading to home")
                                if dx >= 0:
                                    if -stat.x + stat.home[0] < 0:
                                        print("got to home")
                                        ser.write(bytes([137, 0, 0, 0, 0]))
                                        reached = True
                                        has_begun = False
                                        goal_calculated = False
                                        turned = False
                                        steer_left = False
                                        left_origin = False
                                else:
                                    if -stat.x + stat.home[0] >= 0:
                                        print("got to home")
                                        ser.write(bytes([137, 0, 0, 0, 0]))
                                        reached = True
                                        has_begun = False
                                        goal_calculated = False
                                        turned = False
                                        steer_left = False
                                        left_origin = False
                                ser.write(bytes([137, 0, 50, 0, 0]))
                        elif (int)(stat.lightBumper[3]) == 1 or (int)(stat.lightBumper[4]) == 1 or (int)(stat.lightBumper[5]) == 1:
                            ser.write(bytes([137, 0, 20, 255, 255]))
                        else:
                            if int(stat.physBumper[3]) == 1:
                                ser.write(bytes([137, 255, 205, 255, 155]))
                            elif int(stat.physBumper[2]) == 1:
                                ser.write(bytes([137, 255, 205, 0, 100]))
                            else:
                                ser.write(bytes([137, 0, 80, 4, 200]))
                elif steer_left is False and edge_done is False:
                    if has_begun is False:
                        if stat.ang > math.pi:
                            if (2*math.pi - stat.ang) < epsilon_t:
                                print("at zero deg")
                                has_begun = True
                                ser.write(bytes([137, 0, 30, 0, 1]))
                                time.sleep(3.5)
                                ser.write(bytes([137, 0, 80, 252, 56]))
                            else:
                                print("turning left")
                                ser.write(bytes([137, 0, 20, 0, 1]))
                        else:
                            if stat.ang < epsilon_t:
                                print("at zero deg")
                                has_begun = True
                                ser.write(bytes([137, 0, 30, 0, 1]))
                                time.sleep(3.5)
                                ser.write(bytes([137, 0, 80, 252, 56]))
                            else:
                                print("turning right")
                                ser.write(bytes([137, 0, 20, 255, 255]))
                        # ser.write(bytes([137, 0, 80, 252, 56]))
                        # has_begun = True
                    elif start_x != None:
                        err = abs(charge_x - stat.x) + abs(stat.y - charge_y)
                        # print("err:", err)
                        if left_origin == False:
                            if err > 1000:
                                print("left origin")
                                left_origin = True
                        if err < 500 and left_origin:
                            print("going to home")
                            if goal_calculated is False:
                                dx = -stat.x + stat.home[0]
                                dy = -stat.y + stat.home[1]
                                theta = math.atan2(dy, dx) + (3*math.pi/2)
                                if theta < 0:
                                    theta += 2*math.pi
                                if theta > 2*math.pi:
                                    theta -= 2*math.pi
                                dt = theta - stat.ang
                                goal_calculated = True
                            elif turned is False:
                                dt = theta - stat.ang
                                if (abs(dt) < epsilon_t):
                                    turned = True
                                if (abs(dt) > math.pi and dt > 0) or (abs(dt) < math.pi and dt < 0):
                                    print("turning right")
                                    ser.write(bytes([137, 0, 20, 255, 255]))
                                else:
                                    print("turning left")
                                    ser.write(bytes([137, 0, 20, 0, 1]))
                            elif reached is False:
                                print("heading to home")
                                if dx >= 0:
                                    if -stat.x + stat.home[0] < 0:
                                        reached = True
                                        has_begun = False
                                        goal_calculated = False
                                        # turned = False
                                        edge_done = True
                                        initiated = False
                                else:
                                    if -stat.x + stat.home[0] >= 0:
                                        reached = True
                                        has_begun = False
                                        goal_calculated = False
                                        # turned = False
                                        edge_done = True
                                        initiated = False
                                ser.write(bytes([137, 0, 50, 0, 0]))
                        elif (int)(stat.lightBumper[0]) == 1 or (int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1:
                            ser.write(bytes([137, 0, 20, 0, 1]))
                        else:
                            if int(stat.physBumper[3]) == 1:
                                ser.write(bytes([137, 255, 205, 255, 155]))
                            elif int(stat.physBumper[2]) == 1:
                                ser.write(bytes([137, 255, 205, 0, 100]))
                            else:
                                ser.write(bytes([137, 0, 80, 252, 56]))
                elif edge_done is True and lawn_done is False:
                    if lawn_initiated is False:
                        print("initiating")
                        # stat.y_min = np.amin(stat.y_arr) 
                        stat.y_min = -1500 #for debug
                        if stat.ang > math.pi/2:
                            ser.write(bytes([137, 0, 20, 255, 255]))
                        else:
                            ser.write(bytes([137, 0, 20, 0, 1]))
                        if abs(stat.ang - (math.pi/2)) < epsilon_t:
                            to_left = True
                            to_right = False
                            lawn_initiated = True
                            on_wall = False
                    else:
                        if (abs(stat.y - stat.y_min) < 150):
                            lawn_done = True
                        else:
                            err = abs(start_x - stat.x) + abs(start_y - stat.y)
                            if (int)(stat.lightBumper) == 0 and on_wall is False:
                                print("moving straight")
                                # ser.write(bytes([137, 0, 100, 0, 0]))
                                ser.write(bytes([137, 0, 128, 0, 0]))
                            elif (int)(stat.lightBumper) != 0 and on_wall is False:
                                print("on wall")
                                on_wall = True
                                if to_left is True:
                                    to_left = False
                                    to_right = True
                                elif to_right is True:
                                    to_left = True
                                    to_right = False
                                goal_y = stat.y - 150
                            elif on_wall is True and (goal_y <= stat.y):
                                if to_left is True:
                                    if (int)(stat.lightBumper[3]) == 1 or (int)(stat.lightBumper[4]) == 1 or (int)(stat.lightBumper[5]) == 1 or (int)(stat.lightBumper[0]) == 1 or (int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1:
                                        ser.write(bytes([137, 0, 20, 255, 255]))
                                    else:
                                        if int(stat.physBumper[3]) == 1:
                                            ser.write(bytes([137, 255, 205, 255, 155]))
                                        elif int(stat.physBumper[2]) == 1:
                                            ser.write(bytes([137, 255, 205, 0, 100]))
                                        else:
                                            ser.write(bytes([137, 0, 80, 4, 200]))
                                else:
                                    if (int)(stat.lightBumper[0]) == 1 or (int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1 or (int)(stat.lightBumper[3]) == 1 or (int)(stat.lightBumper[4]) == 1 or (int)(stat.lightBumper[5]) == 1:
                                        ser.write(bytes([137, 0, 20, 0, 1]))
                                    else:
                                        if int(stat.physBumper[3]) == 1:
                                            ser.write(bytes([137, 255, 205, 255, 155]))
                                        elif int(stat.physBumper[2]) == 1:
                                            ser.write(bytes([137, 255, 205, 0, 100]))
                                        else:
                                            ser.write(bytes([137, 0, 80, 252, 56]))
                            else:
                                if to_left is True:
                                    if (stat.ang <= (math.pi/2)):
                                        on_wall = False
                                    else:
                                        ser.write(bytes([137, 0, 20, 255, 255]))
                                else:
                                    if (stat.ang >= (3*math.pi/2)):
                                        on_wall = False
                                    else:
                                        ser.write(bytes([137, 0, 20, 0, 1]))
                else:
                    done = True

def main():
    # plt.subplot(1,2,1)
    # data = {'x_data': stat.x_arr, 'y_data': stat.y_arr}
    # # plt.plot('x_data','y_data',data=data)
    # plt.scatter('x_data','y_data',data=data, s=2)
    # plt.axis('scaled')
    # # print(data)
    # print(stat.x, " ", stat.y, " ", stat.ang)
    
    # createMap(50)
    
    dmap = np.loadtxt("map.txt", dtype=int)
    cmap = colors.ListedColormap(['white','gray', 'black'])
    # plt.subplot(1,2,2)
    plt.pcolor(dmap,cmap=cmap,edgecolors='k', linewidths=0.5)
    plt.show()

if __name__ == "__main__":
    main()