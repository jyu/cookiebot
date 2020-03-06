import serial
import time
import keyboard
import threading
import math
import matplotlib.pyplot as plt
import numpy as np
import asyncio
import websockets
import multiprocessing

class Robot_stat:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.dist = 0
        self.ang = 0
        self.ser = serial.Serial("COM5", baudrate=115200, timeout=0.5, writeTimeout=0)
        self.state = ""
        self.x_arr = np.array([0.0])
        self.y_arr = np.array([0.0])
        self.lightBumper = ""
        self.physBumper = ""
    def update(self, x, y, dist, ang, lightBumper, physBumper):
        self.x = x
        self.y = y
        self.dist = dist
        self.ang = ang
        if (self.ang < 0):
            self.ang = (2*math.pi) - self.ang
        if (self.ang > 2*math.pi):
            self.ang = self.ang - (2*math.pi)
        self.x_arr = np.append(self.x_arr,[x])
        self.y_arr = np.append(self.y_arr,[y])
        self.lightBumper = (bin(lightBumper)[2:]).zfill(6)
        self.physBumper = (bin(physBumper)[2:]).zfill(4)
    def print_stat(self):
        print("dist:", self.dist, " ang:", self.ang, " x:", self.x, " y:", self.y)
        # print("bump:", self.lightBumper)

def odometry_fn(stat):
    ser = stat.ser
    dist = 0
    ang = 0
    x = 0
    y = 0
    prev_encL = None
    prev_encR = None
    while True:
        # ser.write(bytes([150, 1]))
        data = None
        pdata1 = pdata2 = pdata3 = pdata4 = 0
        ser.in_waiting
        found = False
        while (found == False):
            data = ser.read(1)
            data = int.from_bytes(data, byteorder='big', signed=False)
            if data == 19: # Header for data packet
                data = ser.read(1)
                data = int.from_bytes(data, byteorder='big', signed=False)
                if data == 10: # N-byte between header and checksum
                    found = True
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

        if prev_encL is None:
            prev_encL = encoderL
            prev_encR = encoderR
        else:
            distL = (encoderL - prev_encL)*math.pi*72.0/508.8
            distR = (encoderR - prev_encR)*math.pi*72.0/508.8
            dist = (distL + distR)/2.0
            if abs(dist) > 1000:
                ser.write(bytes([137, 0, 0, 0, 0]))
                print("fucked up value received")
                ser.reset_output_buffer()
                ser.reset_input_buffer()
            else:
                ang += (distR - distL)/235.0
                prev_encL = encoderL
                prev_encR = encoderR

                x -= dist*math.sin(ang)
                y += dist*math.cos(ang)

                stat.update(x, y, dist, ang, lightBumper, physBumper)
        # stat.print_stat()

def drive_fn(stat):
    ser = stat.ser
    while True:
        if (keyboard.is_pressed('up')):
            ser.write(bytes([137, 0, 100, 0, 0]))
        elif (keyboard.is_pressed('down')):
            ser.write(bytes([137, 255, 156, 0, 0]))
        elif (keyboard.is_pressed('right')):
            ser.write(bytes([137, 0, 100, 255, 255]))
        elif (keyboard.is_pressed('left')):
            ser.write(bytes([137, 0, 100, 0, 1]))
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

def mapping_fn(stat):
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
    ser.write(bytes([137, 0, 100, 4, 0])) # Steer left / Full speed
    # ser.write(bytes([137, 0, 100, 0, 0]))

    #For debug
    # start_x = stat.x
    # start_y = stat.y
    # edge_done = True

    while done is False:
        # First edge following with steering left
        if steer_left is True and edge_done is False:
            if start_x != None:
                err = abs(start_x - stat.x) + abs(start_y - stat.y)
                print("err:", err)
                if left_origin == False:
                    if err > origin_err_goal:
                        left_origin = True
                if err < origin_err_goal - 50 and left_origin:
                    print("returned to origin")
                    ser.write(bytes([137, 0, 0, 0, 0]))
                    ser.write(bytes([137, 0, 100, 0, 1]))
                    time.sleep(3)
                    steer_left = False
                    ser.write(bytes([137, 0, 0, 0, 0]))
                    time.sleep(2)
                    left_origin = False
                    start_x = stat.x
                    start_y = stat.y

            if (int)(stat.lightBumper[3]) == 1 or (int)(stat.lightBumper[4]) == 1:
                ser.write(bytes([137, 0, 100, 255, 255]))
                while (int)(stat.lightBumper[3]) == 1 or (int)(stat.lightBumper[4]) == 1:
                    print("turning")
                if start_x == None:
                    start_x = stat.x
                    start_y = stat.y
                ser.write(bytes([137, 0, 0, 0, 0])) # Is this necessary
            else:
                ser.write(bytes([137, 0, 100, 4, 0]))
                if int(stat.physBumper[3]) == 1:
                    if start_x == None:
                        start_x = stat.x
                        start_y = stat.y
                    print("right bumped")
                    ser.write(bytes([137, 255, 156, 0, 0]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 255, 255]))
                    time.sleep(1)
                    ser.write(bytes([137, 0, 0, 0, 0]))
                elif int(stat.physBumper[2]) == 1:
                    if start_x == None:
                        start_x = stat.x
                        start_y = stat.y
                    print("left bumped")
                    ser.write(bytes([137, 255, 156, 0, 0]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 255, 255]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 0, 0, 0]))
            print("steering left")
        elif steer_left is False and edge_done is False:
            # Second edge following with right steering
            err = abs(start_x - stat.x) + abs(start_y - stat.y)
            print("err:", err)
            if left_origin == False:
                if err > origin_err_goal:
                    left_origin = True
            if err < origin_err_goal - 50 and left_origin:
                print("returned to origin")
                ser.write(bytes([137, 0, 0, 0, 0]))
                edge_done = True
                time.sleep(2)
                start_x = stat.x
                start_y = stat.y
                left_origin = False
            if (int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1:
                ser.write(bytes([137, 0, 100, 0, 1]))
                while (int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1:
                    print("turning")
                ser.write(bytes([137, 0, 0, 0, 0])) # Is this necessary
            else:
                ser.write(bytes([137, 0, 100, 252, 0]))
                if int(stat.physBumper[3]) == 1:
                    print("right bumped")
                    ser.write(bytes([137, 255, 156, 0, 0]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 0, 1]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 0, 0]))
                elif int(stat.physBumper[2]) == 1:
                    print("left bumped")
                    ser.write(bytes([137, 255, 156, 0, 0]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 0, 1]))
                    time.sleep(1)
                    ser.write(bytes([137, 0, 100, 0, 0]))
                print("steering right")
        else:
            # Scanning the rest of the room
            print("scanning the rest")
            err = abs(start_x - stat.x) + abs(start_y - stat.y)
            print("err:", err)
            if left_origin == False:
                if err > origin_err_goal:
                    left_origin = True
            # if err < 100 and left_origin:
            #     print("returned to origin")
            #     ser.write(bytes([137, 0, 0, 0, 0]))
            #     break
            # going to the first wall
            if first_wall is False:
                if (int)(stat.lightBumper[1]) == 0 or (int)(stat.lightBumper[2]) == 0:
                    ser.write(bytes([137, 0, 100, 0, 0]))
                    while (int)(stat.lightBumper[1]) == 0 or (int)(stat.lightBumper[2]) == 0:
                        if int(stat.physBumper[3]) == 1:
                            # print("right bumped")
                            ser.write(bytes([137, 255, 156, 0, 0]))
                            time.sleep(0.2)
                            ser.write(bytes([137, 0, 100, 0, 1]))
                            time.sleep(0.2)
                            ser.write(bytes([137, 0, 100, 0, 0]))
                        elif int(stat.physBumper[2]) == 1:
                            # print("left bumped")
                            ser.write(bytes([137, 255, 156, 0, 0]))
                            time.sleep(0.2)
                            ser.write(bytes([137, 0, 100, 255, 255]))
                            time.sleep(0.2)
                            ser.write(bytes([137, 0, 100, 0, 0]))
                        print("going straight")
                    ser.write(bytes([137, 0, 0, 0, 0])) # Is this necessary
                    time.sleep(1)
                    first_wall = True
                # if (int)(stat.lightBumper[0]) == 1: # Maybe stricter condition required
                #     ser.write(bytes([137, 0, 100, 0, 1]))
                #     while (int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1:
                #         print("turning")
                #     # ser.write(bytes([137, 0, 100, 0, 0]))
                #     ser.write(bytes([137, 0, 0, 0, 0]))
                #     return
                # time.sleep(1)
                ser.write(bytes([137, 0, 100, 0, 1]))
                while (int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1 or (int)(stat.lightBumper[3]) == 1:
                    print("turning")
                hor_ang = stat.ang
                print(hor_ang)
            # hor_ang = (stat.ang - math.pi)%(2*math.pi)
            ser.write(bytes([137, 0, 100, 0, 0]))
            while (int)(stat.lightBumper[2]) == 0 and (int)(stat.lightBumper[3]) == 0: #and \
                #   (int)(stat.lightBumper[1]) == 0 and (int)(stat.lightBumper[4]) == 0:
                if int(stat.physBumper[3]) == 1:
                    print("right bumped")
                    ser.write(bytes([137, 255, 156, 0, 0]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 0, 1]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 0, 0]))
                elif int(stat.physBumper[2]) == 1:
                    print("left bumped")
                    ser.write(bytes([137, 255, 156, 0, 0]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 255, 255]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 0, 0]))
                print("going straight")
            ser.write(bytes([137, 0, 0, 0, 0])) 
            time.sleep(1)   
            print("time to turn")
            if on_right is True:
                ser.write(bytes([137, 0, 100, 0, 1]))
            else:
                ser.write(bytes([137, 0, 100, 255, 255]))
            # while ((int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1 or (int)(stat.lightBumper[3]) == 1) or \
                    # ((int)(stat.lightBumper[0] == 0 and (int)(stat.lightBumper[5] == 0))):
            while (int)(stat.lightBumper[1]) == 1 or (int)(stat.lightBumper[2]) == 1 or (int)(stat.lightBumper[3]) == 1:
                print("turning")
            time.sleep(0.5)
            ser.write(bytes([137, 0, 100, 0, 0]))
            t_end = time.time() + 5
            while time.time() < t_end:
                if int(stat.physBumper[3]) == 1:
                    print("right bumped")
                    ser.write(bytes([137, 255, 156, 0, 0]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 0, 1]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 0, 0]))
                elif int(stat.physBumper[2]) == 1:
                    print("left bumped")
                    ser.write(bytes([137, 255, 156, 0, 0]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 255, 255]))
                    time.sleep(0.2)
                    ser.write(bytes([137, 0, 100, 0, 0]))
                print("moving a bit further")
            
            target_ang = hor_ang if not on_right else (hor_ang - math.pi)%(2 * math.pi)
            if on_right is True:
                ser.write(bytes([137, 0, 100, 0, 1]))
            else:
                ser.write(bytes([137, 0, 100, 255, 255]))
            while abs(stat.ang - target_ang) > epsilon:
                print("target:", target_ang, " err:", abs(stat.ang - target_ang), " or:", on_right)
            ser.write(bytes([137, 0, 0, 0, 0]))
            on_right = not on_right


        # print("R:", right, " FR:", frontRight, " CR:", centerRight, " CL:", centerLeft, " FL:", frontLeft, " L:", left)
        # print("L:", leftBump, " R:", rightBump)
    
def main():
    robot = Robot_stat()
    t_odometry = threading.Thread(target=odometry_fn, args=(robot,), daemon=True)
    t_drive = threading.Thread(target=drive_fn, args=(robot,), daemon=True)
    t_map = threading.Thread(target=mapping_fn, args=(robot,), daemon=True)
    # t_wake = threading.Thread(target=wake_fn, args=(robot,), daemon=True)
    ser = robot.ser
    # Send "Start" Opcode to start Open Interface, Roomba in Passive Mode
    ser.write(bytes([128]))
    # ser.write(bytes([7]))
    # time.sleep(5)
    # ser.write(bytes([128]))
    robot.state = "passive"
    # Send "Safe Mode" Opcode to enable Roomba to respond to commands
    ser.write(bytes([131])) #132:full 131:safe
    # ser.write(bytes([148, 4, 19, 20, 45, 7])) # Initiate streaming
    ser.write(bytes([148, 4, 43, 44, 45, 7])) # Initiate streaming
    robot.state = "safe"
    t_odometry.start()
    t_drive.start()
    # t_map.start()
    # t_wake.start()
    while True:
        # print(ser.in_waiting)
        # print(t_map.is_alive() is False)
        if keyboard.is_pressed("esc"):
            ser.write(bytes([137, 0, 0, 0, 0]))
            ser.write(bytes([150, 0]))
            ser.write(bytes([128])) #return to passive mode
            # while ser.in_waiting:
                # print(ser.in_waiting)
            data = {'x_data': robot.x_arr, 'y_data': robot.y_arr}
            # plt.plot('x_data','y_data',data=data)
            plt.scatter('x_data','y_data',data=data, s=2)
            plt.axis('scaled')
            # print(data)
            break
        elif keyboard.is_pressed("m") and (t_map.is_alive() is False):
            print("map thread button pressed")
            t_map.start()
    plt.show()

if __name__ == "__main__":
    main()
