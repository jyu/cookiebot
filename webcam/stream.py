import cv2
import base64
import asyncio
from websocket import create_connection
import numpy as np
import time
import argparse
import _thread

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-name', default="video.mp4", help="Name of video") # Video name
args = parser.parse_args()

# Out file, defaults to video.mp4
name = args.name

#cap = cv2.VideoCapture(0)
# USB webcam
cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)

width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(name, fourcc, 30.0, (width, height))

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 2000, 1000)
black = (255,255,255)
font = cv2.FONT_HERSHEY_SIMPLEX

pos = [""]
def get_pos(frame, ws, pos):
  _, buf = cv2.imencode('.jpg', frame)
  jpg_as_text = base64.b64encode(buf)
  ws.send(jpg_as_text)
  
  # For streaming point
  pos[0] = ws.recv()


time.sleep(5)    

uri = "ws://72.74.137.57:5000"
count = 0
start = time.time()
ws = create_connection(uri)
while True:
  success, frame = cap.read()

  if count % 10 == 0:
    _thread.start_new_thread(get_pos, (frame, ws, pos))

  out.write(frame)
  cv2.putText(frame, str(pos[0]), (20, 40), font, 1, black, 2, cv2.LINE_AA)
  
  cv2.imshow('image', frame)

  # For streaming openpose img
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  count += 1
  end = time.time()
  print("FPS", count / (end - start), end="\r")

out.release()
ws.close()
