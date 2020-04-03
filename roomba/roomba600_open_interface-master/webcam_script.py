import cv2
import base64
import asyncio
import websockets
import numpy as np
import time
import argparse
import sys
import math
import keyboard

def capture():
  # Local webcam
  cap = cv2.VideoCapture(0)

  # Probably windows USB webcam
  # cap = cv2.VideoCapture(1) 

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
  name = "HumanRobotInteraction.mp4"
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
      # cv2.putText(frame, str(round(left,2)) + " time left", (60, 60), font, 2, black, 1, cv2.LINE_AA)
      if left <= 0:
        waiting = False
    # elif total_time < elapsed:
    #   break
    else:
      print()
      out.write(frame)
    # cv2.imshow('frame', frame)
    # Press q to exit
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    if keyboard.is_pressed("q"):
      break
  print("cap ended")
  cap.release()
  out.release()
  return

def main():
  # cap_begun = False
  # while True:
  #   if keyboard.is_pressed("c") and cap_begun is False:
  #     print("cap begun")
  #     cap_begun = True
  #     capture()
  #     return
  # video = cv2.VideoCapture('MoveItMoveIt.mp4') 
  # total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
  # black = (0,0,0)
  # font = cv2.FONT_HERSHEY_SIMPLEX
  # coord_list = np.array([])
  # width = int(video.get(3))
  # height = int(video.get(4))
  # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  # out = cv2.VideoWriter("result.mp4", fourcc, 30.0, (width, height))

  # count = 0
  # with open("test.txt") as f:
  #   # for line in f:
  #   line = f.readline()
  #   while line:
  #     # print(line)
  #     if count > 75:
  #       coord_list = np.append(coord_list, line)
  #     line = f.readline()
  #     count+=1

  # length = np.size(coord_list)
  # # length = count
  # # print(length)
  # success, frame = video.read()
  # # print(success)

  # i = 0.0
  # inter = length/total_frames
  # print(length)
  # print(total_frames)
  # print(inter)
  # while (success is True):
  #   if round(i) < np.size(coord_list):
  #     cv2.putText(frame, str(coord_list[round(i)]), (60, 60), font, 0.5, black, 1, cv2.LINE_AA)
  #   i+=length/total_frames
  #   # cv2.imshow('video', frame) 
  #   out.write(frame)
  #   success, frame = video.read()
  # video.release()
  # out.release()
  capture()
  return
  
if __name__ == "__main__":
    main()
  