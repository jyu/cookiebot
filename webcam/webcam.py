import cv2
import base64
from websocket import create_connection

cap = cv2.VideoCapture(0)
#ws = create_connection("ws://72.74.137.57:5555")
print("Trying to connect")
ws = create_connection("ws://72.74.137.57:5000")
print("connected")

while True:
  success, frame = cap.read()
  cv2.imshow('frame', frame)
  _, buf = cv2.imencode('.jpg', frame)
  print('buf type', type(buf))
  print('buf', buf)
  jpg_as_text = base64.b64encode(buf)
  print('jpg as text', jpg_as_text)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
