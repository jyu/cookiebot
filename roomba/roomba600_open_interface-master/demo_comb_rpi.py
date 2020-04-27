import serial
import time
# import keyboard
import threading
import math
import matplotlib.pyplot as plt
import numpy as np
# import asyncio
# import websockets
# from multiprocessing import Process
from matplotlib import colors
# import webcam_script
# import cv2
import base64
# import asyncio
# import websockets
import sys
# from pynput import keyboard

update_interval = 1

class Robot_stat:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.dist = 0
        self.ang = 0
        # self.ser = serial.Serial("COM5", baudrate=115200, timeout=55.5, writeTimeout=55.5)
        self.ser = serial.Serial("COM8", baudrate=115200, timeout=None, writeTimeout=0)
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

def cell2XY(cell_x, cell_y, x_min, y_min, cell_size):
    x = x_min
    y = y_min
    x += cell_x * cell_size
    y += cell_y * cell_size
    x += (cell_size/2)
    x += (cell_size/2)
    return (x, y)

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
    # dt = - stat.ang
    # if (abs(dt) > math.pi and dt > 0) or (abs(dt) < math.pi and dt < 0):
    #     print("turning right")
    #     ser.write(bytes([137, 0, 20, 255, 255]))
    #     while(abs(dt) > epsilon_t):
    #         dt = -stat.ang
    #         time.sleep(0)
    # else:
    #     print("turning left")
    #     ser.write(bytes([137, 0, 20, 0, 1]))
    #     while(abs(dt) > epsilon_t):
    #         dt = -stat.ang
    #         time.sleep(0)
    # print("at home")
    ser.write(bytes([137, 0, 0, 0, 0]))
    return

class Node:
    def __init__(self, cell_x, cell_y):
        self.cell_x = cell_x
        self.cell_y = cell_y
        self.f = 0
        self.g = sys.maxsize
        self.h = 0
        self.parent = None
    def __eq__(self, other):
        return (self.__class__ == other.__class__ and self.cell_x == other.cell_x and self.cell_y == other.cell_y)

fig_count = 0

def find_path(goal):
    global fig_count
    global stat
    fig_count += 1
    map = np.loadtxt("map.txt", dtype=int)
    x1 = np.loadtxt("x_arr_e1.txt")
    x2 = np.loadtxt("x_arr_e2.txt")
    x1_min = np.min(x1)
    x2_min = np.min(x2)
    x_min = min(x1_min, x2_min)
    y1 = np.loadtxt("y_arr_e1.txt")
    y2 = np.loadtxt("y_arr_e2.txt")
    y1_min = np.min(y1)
    y2_min = np.min(y2)
    y_min = min(y1_min, y2_min)
    y_size = map.shape[0]
    x_size = map.shape[1]
    cell_size = 50
    # goal = (0, -300)
    # curr = (100, -1000)
    curr = (stat.x, stat.y)
    (goal_cell_x, goal_cell_y) = XY2Cell(goal[0], goal[1], x_min, y_min, cell_size)
    (curr_cell_x, curr_cell_y) = XY2Cell(curr[0], curr[1], x_min, y_min, cell_size)
    map[goal_cell_y, goal_cell_x] = 3
    map[curr_cell_y, curr_cell_x] = 4
    open_list = []
    closed_list = []
    initial_node = Node(curr_cell_x, curr_cell_y)
    initial_node.g = 0
    open_list.append(initial_node)
    curr_node = initial_node
    count = 0
    goal_node = None
    # print("curr:", initial_node.cell_x, " ", initial_node.cell_y, " goal:", goal_cell_x, " ", goal_cell_y)
    while (len(open_list) > 0):
        count+=1
        minf = sys.maxsize
        for node_tmp in open_list:
            if node_tmp.f < minf:
                curr_node = node_tmp
                minf = node_tmp.f
        # neighbor_list = []
        closed_list.append(curr_node)
        open_list.remove(curr_node)
        if (curr_node.cell_y == goal_cell_y) and (curr_node.cell_x == goal_cell_x):
            goal_node = curr_node
            break
        for i in range(-1,2):
            for j in range(-1,2):
                if (curr_node.cell_x+j >= 0) and (curr_node.cell_x+j < x_size) and \
                    (curr_node.cell_y+i >= 0) and (curr_node.cell_y+i < y_size):
                    if (map[curr_node.cell_y+i, curr_node.cell_x+j] < 2):
                        newNode = Node(curr_node.cell_x+i, curr_node.cell_y+j)
                        if (newNode in open_list):
                            index = open_list.index(newNode)
                            newNode = open_list[index]
                            # open_list.remove(newNode)
                        if (newNode not in closed_list):
                            if (newNode.g > curr_node.g + map[curr_node.cell_y+i, curr_node.cell_x+j]):
                                newNode.g = curr_node.g + map[curr_node.cell_y+i, curr_node.cell_x+j] + 1 #math.sqrt(i**2 + j**2) #abs(i) + abs(j)
                                newNode.parent = curr_node
                                h_new = 0
                                h_new += abs(goal_cell_x - (curr_node.cell_x+j))
                                h_new += abs(goal_cell_y - (curr_node.cell_y+i))
                                newNode.h = h_new
                                newNode.f = newNode.g + newNode.h
                                newNode.parent = curr_node
                                if (newNode not in open_list):
                                    open_list.append(newNode)

    node_tmp = goal_node
    if (goal_node is None):
        print("no path found")
    path = []
    while (node_tmp.parent is not None):
        path.append((node_tmp.cell_x, node_tmp.cell_y))
        if (map[node_tmp.cell_y, node_tmp.cell_x] < 2):
            map[node_tmp.cell_y, node_tmp.cell_x] = 5
        node_tmp = node_tmp.parent
    path.reverse()
    # print(path)
    path_xy = []
    for coord in path:
        (x,y) = cell2XY(coord[0], coord[1], x_min, y_min, cell_size)
        path_xy.append((x,y))
    # print(path_xy)
    cmap = colors.ListedColormap(['white','gray','black','blue','red','green'])
    plt.pcolor(map,cmap=cmap,edgecolors='k', linewidths=0.5)
    fname = "path" + str(fig_count) + ".png"
    plt.savefig(fname)
    # plt.show()
    return path_xy

def main():
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
    go_home = False
    coord_cnt = 0
    coord_done = False
    drive_done = False
    path_found = False
    done_turning = False
    path = []
    d_flag_pos = False
    d_flag_neg = False
    start = time.time()
    start_test = time.time()
    # move_back = False
    e1_flag = False
    e2_flag = False
    e3_flag = False

    # time.sleep(3)
    # ser.write(bytes([137, 255, 156, 128, 0]))
    # time.sleep(2)
    moved_back = False
    go_home_command = False
    turn_zero = False
    data_ms = np.array([])


    while True:
        curr = time.time()
        curr_test = time.time()
        if (curr - start) > 5 and go_home_command is False and e3_flag is False:
            go_home_command = True
        else:
            go_home_command = False
        data_ms = np.append(data_ms, start_test - curr)
        # if (curr - start) > 5 (e1_flag is False):
        #     move_back = True
        #     e1_flag = True
        # elif (curr - start) > 7 and (curr - start) < 10:
        #     ser.write(bytes([137, 0, 0, 0, 0]))
        # elif (curr - start) > 10 and (e2_flag is False):
        #     go_home = True
        #     path_goal = (0, -680)
        #     e2_flag = True
        # elif and (drive_done is True) and (e3_flag is False):
        #     drive_done = False

        # if keyboard.is_pressed("esc") or done is True:
        if done is True:
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
            ser.close()
            
            createMap(50)
            cmap = colors.ListedColormap(['white','gray', 'black'])
            plt.subplot(1,2,2)
            plt.pcolor(stat.map,cmap=cmap,edgecolors='k', linewidths=0.5)
            plt.show()
            # np.savetxt('x_arr.txt', stat.x_arr, delimiter=',')
            # np.savetxt('y_arr.txt', stat.y_arr, delimiter=',')
            # np.savetxt('map.txt', stat.map, fmt='%s')
            return
            
        # if (keyboard.is_pressed('up')):
        #     ser.write(bytes([137, 0, 100, 128, 0]))
        # elif (keyboard.is_pressed('down')) or (moved_back is False):
        elif moved_back is False:
            if moved_back is False:
                if stat.y < -300:
                    moved_back = True
                    ser.write(bytes([137, 0, 0, 0, 0]))
                else:
                    ser.write(bytes([137, 255, 206, 128, 0]))
            else:
                ser.write(bytes([137, 255, 156, 128, 0]))
        # elif (keyboard.is_pressed('right')):
        #     ser.write(bytes([137, 0, 20, 255, 255]))
        # elif (keyboard.is_pressed('left')):
        #     ser.write(bytes([137, 0, 20, 0, 1]))
        # elif (keyboard.is_pressed('space')):
        #     ser.write(bytes([137, 0, 0, 0, 0]))
        # elif (keyboard.is_pressed('d')):
        #     ser.write(bytes([143])) #dock
        # elif (keyboard.is_pressed('m')):
        #     map_start = True
        # elif (keyboard.is_pressed('h')):
        elif go_home_command is True:
            if e1_flag is False and go_home is False:
                print("goal 1 set")
                # path_goal = (0, -680)
                path_goal = (370, -1945)
                e1_flag = True
                drive_done = False
                # go_home = True
            elif e2_flag is False and go_home is False:
                print("goal 2 set")
                # path_goal = (365, -1965)
                path_goal = (-200, -680)
                e2_flag = True
                drive_done = False
                # go_home = True
            elif e3_flag is False and go_home is False:
                print("goal 3 set")
                path_goal = (0, -350)
                e3_flag = True
                drive_done = False
                # go_home = True
            go_home = True

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
                # ser.write(bytes([137, 0, 0, 0, 0]))
                print("fucked up value received") 
                # return
                # done = True
                encoderL = prev_encL
                encoderR = prev_encR
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
                # print(stat.x, stat.y)
        if go_home is True and drive_done is False:
            if path_found is False:
                # path = find_path((0, -300))
                path = find_path(path_goal)
                path_found = True
            # for coord in path:
            if (coord_cnt < len(path)):
                coord = path[coord_cnt]
            if coord_done is False:
                # print(coord_cnt, "/", len(path))
                # drive2(coord[0], coord[1])
                dx = -stat.x + coord[0]
                dy = -stat.y + coord[1]
                theta = math.atan2(dy, dx) + (3*math.pi/2)
                if theta < 0:
                    theta += 2*math.pi
                if theta > 2*math.pi:
                    theta -= 2*math.pi
                dt = theta - stat.ang
                # print("dx:", dx, " dy:", dy, " curr:", stat.ang, " goal theta:", theta, " dt:",dt)

                epsilon_t = 0.006
                epsilon_d = 20
                if done_turning is False:
                    if (abs(dt) > math.pi and dt > 0) or (abs(dt) < math.pi and dt < 0):
                        # print("turning right")
                        if (abs(dt) > epsilon_t):
                            ser.write(bytes([137, 0, 20, 255, 255]))
                        else:
                            done_turning = True
                            # print("done turning")
                        # while(abs(dt) > epsilon_t):
                            # dt = theta - stat.ang
                            # time.sleep(0)
                    else:
                        # print("turning left")
                        if (abs(dt) > epsilon_t):
                            ser.write(bytes([137, 0, 20, 0, 1]))
                        else:
                            done_turning = True
                            # print("done turning")
                        # while(abs(dt) > epsilon_t):
                        #     dt = theta - stat.ang
                        #     time.sleep(0)
                # print("done turning")
                else:
                    # print("goal_coor:", coord, " curr:", stat.x,",", stat.y, " df", d_flag_neg, " ", d_flag_pos)
                    # if coord[0] > stat.x:
                    if d_flag_pos is True:
                        if (coord[0] - stat.x) > 0 and (abs(abs(coord[1]) - abs(stat.y)) > 0.5):
                            ser.write(bytes([137, 0, 50, 0, 0]))
                        else:
                            # print("reached goal")
                            ser.write(bytes([137, 0, 0, 0, 0]))
                            coord_done = True
                            d_flag_pos = False
                            d_flag_neg = False
                    # else:
                    elif d_flag_neg is True:
                        if (stat.x - coord[0]) > 0 and (abs(abs(coord[1]) - abs(stat.y)) > 0.5):
                            ser.write(bytes([137, 0, 50, 0, 0]))
                        else:
                            # print("reached goal")
                            ser.write(bytes([137, 0, 0, 0, 0]))
                            coord_done = True
                            d_flag_neg = False
                            d_flag_pos = False
                    elif d_flag_neg is False and d_flag_pos is False:
                        # print("setting flag")
                        if coord[0] > stat.x:
                            d_flag_pos = True
                        else:
                            d_flag_neg = True
            else:
                if e3_flag is True and turn_zero is True: #coord_cnt == len(path):
                    print("turning to home")
                    dt = - stat.ang
                    epsilon_t = 0.006
                    if (abs(dt) > math.pi and dt > 0) or (abs(dt) < math.pi and dt < 0):
                        # print("turning right")
                        if (abs(dt) > epsilon_t):
                            ser.write(bytes([137, 0, 20, 255, 255]))
                        else:
                            e3_flag = False
                            # print("done turning")
                            print("e3 done")
                            ser.write(bytes([143]))
                            path_found = False
                            done = True
                    else:
                        # print("turning left")
                        if (abs(dt) > epsilon_t):
                            ser.write(bytes([137, 0, 20, 0, 1]))
                        else:
                            e3_flag = False
                            # print("done turning")
                            print("e3 done")
                            ser.write(bytes([143]))
                            path_found = False
                            done = True
                else:
                    ser.write(bytes([137, 0, 0, 0, 0]))
                    coord_cnt += 1
                    print(coord_cnt, "/", len(path))
                    if coord_cnt == len(path):
                        print("path done")
                        if e3_flag is False:
                            drive_done = True
                            go_home = False
                            path_found = False
                            go_home_command = False
                            done_turning = False
                            coord_done = False
                            coord_cnt = 0
                            d_flag_neg = False
                            d_flag_pos = False
                            start = time.time()
                        else:
                            # drive_done = True
                            # go_home = False
                            # path_found = False
                            # go_home_command = False
                            # done_turning = False
                            # coord_done = False
                            # coord_cnt = 0
                            # d_flag_neg = False
                            # d_flag_pos = False
                            # start = time.time()
                            coord_cnt -= 1
                            turn_zero = True
                    else:
                        coord_done = False
                        done_turning = False
                        d_flag_neg = False
                        d_flag_pos = False




if __name__ == "__main__":
    main()