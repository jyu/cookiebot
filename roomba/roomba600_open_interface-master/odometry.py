import serial
import time
import keyboard
import threading
import math
import matplotlib.pyplot as plt
import numpy as np
import asyncio
import websockets

ser = serial.Serial("COM5", baudrate=115200, timeout=0.5)

# Send "Start" Opcode to start Open Interface, Roomba in Passive Mode
ser.write(bytes([128]))
# Send "Safe Mode" Opcode to enable Roomba to respond to commands
ser.write(bytes([131])) #132:full 131:safe

x = y = dist = ang = 0.0
x_arr = np.array([0.0])
y_arr = np.array([0.0])
tmp = 0
prev_encoderL = prev_encoderR = 0

ser.write(bytes([148, 2, 19, 20])) # request data packet


while True:
    #ser.write(bytes([132]))
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
    elif (keyboard.is_pressed('q')):
        ser.write(bytes([137, 0, 0, 0, 0]))
        break

    """
    ser.write(bytes([142, 19]))
    ser.in_waiting
    dist_b = ser.read(2)
    dist = int.from_bytes(dist_b, byteorder='big', signed=True)

    ser.write(bytes([142, 20]))
    ser.in_waiting
    ang_b = ser.read(2)
    ang -= int.from_bytes(ang_b, byteorder='big', signed=True)
    """
    
    ser.in_waiting
    found = False
    while (found == False):
        data = ser.read(1)
        data = int.from_bytes(data, byteorder='big', signed=False)
        if data == 19:
            data = ser.read(1)
            data = int.from_bytes(data, byteorder='big', signed=False)
            # print("2",hex(data))
            if data == 6:
                found = True

    if found:
        # print("found")
        ser.read(1)
        pdata1 = ser.read(2)
        dist = int.from_bytes(pdata1, byteorder='big', signed=True)
        ser.read(1)
        pdata2 = ser.read(2)
        ang -= int.from_bytes(pdata2, byteorder='big', signed=True)
        ser.read(1)
        # print("dist:", pdata1, " ang:", pdata2)

    found = False
    ser.reset_input_buffer()

    # ser.write(bytes([142, 43]))
    # ser.in_waiting
    # encoderL_b = ser.read(2)
    # encoderL = int.from_bytes(encoderL_b, byteorder='big', signed=True)
    # dL = encoderL - prev_encoderL

    # ser.write(bytes([142, 44]))
    # ser.in_waiting
    # encoderR_b = ser.read(2)
    # encoderR = int.from_bytes(encoderR_b, byteorder='big', signed=True)
    # dR = encoderR - prev_encoderR

    # print('prevL:', prev_encoderL, ' currL:', encoderL, ' dL:', dL, ' dR:', dR)
    # if tmp == 0 :
        # dL = dR = 0
    
    # Invalid data read
    if dist > 100 or tmp == 0:
        dist = 0
        ang = 0

    # prev_encoderL = encoderL
    # prev_encoderR = encoderR

    # distL = dL * (math.pi * 72.0/508.8)
    # distR = dR * (math.pi * 72.0/508.8)
    # dist = ((distL + distR)/2.0) - dist
    # ang += (distL - distR)/235.0

    #print('dist:',dist, ' ang:', ang)
    # x += dist*math.sin(ang)
    x += dist*math.sin(math.radians(ang))
    y += dist*math.cos(math.radians(ang))
    
    print('dist:',dist/1000, ' ang:', ang, ' x:', x/1000, ' y:', y/1000)
    if (1):
        x_arr = np.append(x_arr,[x])
        y_arr = np.append(y_arr,[y])
    tmp += 1
# print(x_arr, y_arr)
data = {'x_data': x_arr, 'y_data': y_arr}
plt.plot('x_data','y_data',data=data)
plt.show()

ser.write(bytes([128])) #return to passive mode

def XY_coordinate(dist, ang, x, y):
    while True:
        ser.in_waiting
        found = False
        while (found == False):
            data = ser.read(1)
            data = int.from_bytes(data, byteorder='big', signed=False)
            if data == 19:
                data = ser.read(1)
                data = int.from_bytes(data, byteorder='big', signed=False)
                # print("2",hex(data))
                if data == 6:
                    found = True

        if found:
            # print("found")
            ser.read(1)
            pdata1 = ser.read(2)
            dist = int.from_bytes(pdata1, byteorder='big', signed=True)
            ser.read(1)
            pdata2 = ser.read(2)
            ang -= int.from_bytes(pdata2, byteorder='big', signed=True)
            ser.read(1)
            # print("dist:", pdata1, " ang:", pdata2)
        