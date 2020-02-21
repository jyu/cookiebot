import serial
import time
import keyboard
import threading

# Open new connection (NOTE, change port)
ser = serial.Serial("COM5", baudrate=115200, timeout=0.5)

# Actuator Commands
# Send "Start" Opcode to start Open Interface, Roomba in Passive Mode
ser.write(bytes([128]))
# Send "Safe Mode" Opcode to enable Roomba to respond to commands
ser.write(bytes([131])) #132:full 131:safe
# Start Brushes
# ser.write(bytes([144,100,100,100]))
# time.sleep(1)
# Stop Brushes
# ser.write(bytes([144,0,0,0]))

# ser.write(bytes([137, 255, 56, 1, 244]))
# time.sleep(1)
# ser.write(bytes([137, 0, 0, 0, 0]))
# ser.write(bytes([143])) #dock
# ser.write(bytes([131]))
# ser.write(bytes([128]))

# play song? doesn't work
# ser.write(bytes([140, 1, 2, 50, 64, 70, 32]))
# ser.write(bytes([141, 1]))
# time.sleep(2)



# while True:
#     try:
#         if (keyboard.is_pressed('up')):
#             ser.write(bytes([137, 0, 200, 0, 0]))
#         elif (keyboard.is_pressed('down')):
#             ser.write(bytes([137, 255, 56, 0, 0]))
#     except:
#         ser.write(bytes([137, 0, 0, 0, 0]))
#         break

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
    elif (keyboard.is_pressed('q')):
        ser.write(bytes([137, 0, 0, 0, 0]))
        break
    # ser.write(bytes([137, 0, 0, 0, 0]))

# Input Commands (Read State / Sensors)
# Ask for Sensor Packed ID 21 (Battery Charging State)
ser.write(bytes([142, 21]))
# returns 1, for single byte in input buffer for Packet Id 21
ser.in_waiting
# read input buffer
tmp = ser.read(1)
# convert byte response to integer
res = int.from_bytes(tmp, byteorder='big', signed=False)
# will return 2 for Full Charging State
print(res)

# time.sleep(0.5)
ser.write(bytes([142, 22]))
ser.in_waiting
voltage = ser.read(2)
res_v = int.from_bytes(voltage, byteorder='big', signed=False)
print(res_v)
# time.sleep(0.5)
# ser.write(bytes([142, 23]))
# ser.in_waiting
# current = ser.read(2)
# res = int.from_bytes(current, byteorder='big', signed=True)
# print(res)

# ser.write(bytes([173])) #stop
ser.write(bytes([128])) #return to passive mode

