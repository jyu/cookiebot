import cv2
import base64
import asyncio
import websockets
import numpy as np
import time
import argparse
import sys
import math

def capture():
  global stat
  # Local webcam
  #cap = cv2.VideoCapture(0)

  # Probably windows USB webcam
  cap = cv2.VideoCapture(1) 

  # Ubuntu USB webcam
  # cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)

  # Parse arguments
  # parser = argparse.ArgumentParser()
  # parser.add_argument('-name', default="video.mp4", help="Name of video") # Video name
  # parser.add_argument('-wait', default=0, help="Seconds to wait before recording") # Wait time
  # parser.add_argument('-length', default=5, help="Video length in seconds") # Video length
  # args = parser.parse_args()

  # Out file, defaults to video.mp4
  # name = args.name
  name = "MoveItMoveIt.mp4"
  # wait_time = int(args.wait)
  wait_time = 10
  # total_time = int(args.length) + wait_time
  # if (int(args.length) <= 0):
  #   total_time = math.inf

  width = int(cap.get(3))
  height = int(cap.get(4))


  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(name, fourcc, 30.0, (width, height))

  start = time.time()

  waiting = False
  if wait_time > 0:
    waiting = True

  while True:
    print("initiate capturing")
    success, frame = cap.read()
    curr = time.time()
    elapsed = round(curr - start, 2)
    
    if waiting:
      print("waiting")
      # end = time.time()
      # elapsed = round(end - start, 2)
      black = (0,0,0)
      font = cv2.FONT_HERSHEY_SIMPLEX
      left = wait_time - elapsed
      cv2.putText(frame, str(round(left,2)) + " time left", (60, 60), font, 2, black, 1, cv2.LINE_AA)
      if left <= 0:
        waiting = False
    # elif total_time < elapsed:
    #   break
    else:
      out.write(frame)
    cv2.putText(frame, )
    cv2.imshow('frame', frame)
    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  out.release()
  return
