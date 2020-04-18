import cv2
import base64
import asyncio
import websockets
import numpy as np
import time
import argparse
import sys

# Local webcam
#cap = cv2.VideoCapture(0)

# Probably windows USB webcam
#cap = cv2.VideoCapture(1) 

# Ubuntu USB webcam
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-name', default="video.mp4", help="Name of video") # Video name
parser.add_argument('-wait', default=0, help="Seconds to wait before recording") # Wait time
parser.add_argument('-length', default=0, help="Seconds to wait before recording") # Video time
args = parser.parse_args()

# Out file, defaults to video.mp4
name = args.name
wait_time = int(args.wait)
length = int(args.length)

width = int(cap.get(3))
height = int(cap.get(4))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(name, fourcc, 30.0, (width, height))

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 2000, 1000)

waiting = False
if wait_time > 0:
  waiting = True

frames = 0 
black = (255,255,255)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
  success, frame = cap.read()
  frames += 1
  
  if waiting:
    time_left = round(wait_time - frames/30, 2)
    cv2.putText(frame, str(time_left) + " time left", (20, 40), font, 1, black, 2, cv2.LINE_AA)
    if time_left <= 0:
      waiting = False
      frames = 0
  else:
    out.write(frame)
    time_left = round(length - frames/30, 2)
    cv2.putText(frame, str(time_left) + " time left recording", (20, 40), font, 1, black, 2, cv2.LINE_AA)
    if time_left < 0:
      break
  
  cv2.imshow('image', frame)
  # Press q to exit
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
out.release()
