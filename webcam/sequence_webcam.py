import cv2
import base64
import asyncio
import websockets
import numpy as np
import time
import argparse
import sys
import os

#seq = [(0,1), (0,2), (-1,0), (-2, 0), (-2, 1), (-2, 2), (-1, 1), (-1, 2)]
seq = [(0,1), (0,2), (1,0), (2, 0), (2, 1), (2, 2), (1, 1), (1, 2)]

# Ubuntu USB webcam
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dir', default="point_videos", help="Name of video dir") # Video name
parser.add_argument('-wait', default=5, help="Seconds to wait before recording") # Wait time
parser.add_argument('-length', default=30, help="Seconds for length of segment") # Recording time
args = parser.parse_args()

# Out file, defaults to video.mp4
out_dir = args.dir
length = int(args.length)
wait = int(args.wait)
wait_time = wait

width = int(cap.get(3))
height = int(cap.get(4))

log = open('log.txt', 'w')
def getVideoWriterForClass(pos):
  x = str(pos[0]).replace('-', 'n')
  y = str(pos[1])
  path = out_dir + '/point_' + x + '_' + y
  
  files = len(os.listdir(path))
  video_path = path + '/' + str(files + 1) + '.mp4'
  print(video_path)
  log.write(video_path + '\n')

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
  return out

i = 0
out = getVideoWriterForClass(seq[i])

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
    cv2.putText(frame, str(seq[i]) + " next position", (20, 80), font, 1, black, 2, cv2.LINE_AA)
    if time_left <= 0:
      waiting = False
      frames = 0
  else:
    out.write(frame)
    time_left = round(length - frames/30, 2)
    cv2.putText(frame, str(time_left) + " time left recording", (20, 40), font, 1, black, 2, cv2.LINE_AA)
    cv2.putText(frame, str(seq[i]) + " position", (20, 80), font, 1, black, 2, cv2.LINE_AA)
    if i < len(seq) - 1:
      cv2.putText(frame, str(seq[i+1]) + " next position", (20, 120), font, 1, black, 2, cv2.LINE_AA)
    else:
      cv2.putText(frame, "done next", (20, 120), font, 1, black, 2, cv2.LINE_AA)
      
    if time_left < 0:
      i += 1
      if i >= len(seq):
        break
      frames = 0
      out.release()
      out = getVideoWriterForClass(seq[i])
      waiting = True
      wait_time = wait
  
  cv2.imshow('image', frame)
  # Press q to exit
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
out.release()
