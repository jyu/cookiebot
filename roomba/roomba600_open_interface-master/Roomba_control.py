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
        self.x_arr = np.append(self.x_arr,[x])
        self.y_arr = np.append(self.y_arr,[y])
        self.lightBumper = (bin(lightBumper)[2:]).zfill(6)
        self.physBumper = (bin(physBumper)[2:]).zfill(4)
    def print_stat(self):
        print("dist:", self.dist, " ang:", self.ang, " x:", self.x, " y:", self.y)
        # print("bump:", self.lightBumper)

def odometry_fn(stat):
    ser = stat.ser
    
    while True:
        # ser.write(bytes([150, 1]))
        data = None
        pdata1 = pdata2 = pdata3 = pdata4 = 0
        dist = stat.dist
        ang = stat.ang
        x = stat.x
        y = stat.y

        ser.in_waiting
        found = False
        while (found == False):
            data = ser.read(1)
            data = int.from_bytes(data, byteorder='big', signed=False)
            if data == 19: # Header for data packet
                data = ser.read(1)
                data = int.from_bytes(data, byteorder='big', signed=False)
                if data == 10: # Header for distance data within the packet
                    found = True
        if found:
            ser.read(1)
            pdata1 = ser.read(2)
            dist = int.from_bytes(pdata1, byteorder='big', signed=True)
            ser.read(1)
            pdata2 = ser.read(2)
            ang -= int.from_bytes(pdata2, byteorder='big', signed=True)
            ser.read(1)
            pdata3 = ser.read(1)
            lightBumper = int.from_bytes(pdata3, byteorder='big', signed=False)
            ser.read(1)
            pdata4 = ser.read(1)
            physBumper = int.from_bytes(pdata4, byteorder='big', signed=False)
            # print(physBumper)
            # ser.read(1)
            # print("dist:", pdata1, " ang:", pdata2)
            # ser.write(bytes([150, 0]))
        """
        if found and ser.in_waiting >= 10:
            data = ser.read(10)
            ser.write(bytes([150, 0]))
            print(data)
        """

        x += dist*math.sin(math.radians(ang))
        y += dist*math.cos(math.radians(ang))

        stat.update(x, y, dist, ang, lightBumper, physBumper)
        # stat.print_stat()
        

def odometry_fn2(stat):
    ser = stat.ser
    while True:
        ang = stat.ang
        x = stat.x
        y = stat.y
        ser.write(bytes([149, 4, 19, 20, 45, 7]))
        ser.in_waiting
        dist = int.from_bytes(ser.read(2), byteorder='big', signed=True)
        ang -= int.from_bytes(ser.read(2), byteorder='big', signed=True)
        lightBumper = int.from_bytes(ser.read(1), byteorder='big', signed=False)
        physBumper = int.from_bytes(ser.read(1), byteorder='big', signed=False)

        x += dist*math.sin(math.radians(ang))
        y += dist*math.cos(math.radians(ang))

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
    # complete = False
    # while complete == False:
    #     while 
    # print("map thread")
    right = frontRight = centerRight = centerLeft = frontLeft = left = 0
    leftBump = rightBump = 0
    while True:
        print() #why is this necessary wtf
        lightBumper = stat.lightBumper
        physBumper = stat.physBumper
        if lightBumper != "":
            right = int(lightBumper[0])
            frontRight = int(lightBumper[1])
            centerRight = int(lightBumper[2])
            centerLeft = int(lightBumper[3])
            frontLeft = int(lightBumper[4])
            left = int(lightBumper[5])
        if physBumper != "":
            rightBump = int(physBumper[3])
            leftBump = int(physBumper[2])
        if rightBump == 1:
            print("right bumped")
            # ser.write(bytes([137, 0, 0, 0, 0]))
            # ser.write(bytes([137, 255, 156, 0, 0]))
            # time.sleep(0.5)
            # ser.write(bytes([137, 0, 100, 0, 1]))
            # time.sleep(0.5)
            # ser.write(bytes([137, 0, 0, 0, 0]))
        # if leftBump == 1:
            # print("right bumped")

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
    robot.state = "passive"
    # Send "Safe Mode" Opcode to enable Roomba to respond to commands
    ser.write(bytes([131])) #132:full 131:safe
    ser.write(bytes([148, 4, 19, 20, 45, 7]))
    robot.state = "safe"
    t_odometry.start()
    t_drive.start()
    t_map.start()
    # t_wake.start()
    while True:
        # print(ser.in_waiting)
        if keyboard.is_pressed("esc"):
            ser.write(bytes([137, 0, 0, 0, 0]))
            ser.write(bytes([150, 0]))
            ser.write(bytes([128])) #return to passive mode
            # while ser.in_waiting:
                # print(ser.in_waiting)
            data = {'x_data': robot.x_arr, 'y_data': robot.y_arr}
            # plt.plot('x_data','y_data',data=data)
            plt.scatter('x_data','y_data',data=data)
            # print(data)
            break
    plt.show()

if __name__ == "__main__":
    main()
