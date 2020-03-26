import cv2
import base64
import asyncio
import websockets
import numpy as np
import time

# Local webcam
#cap = cv2.VideoCapture(0)
# USB webcam
cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)

async def hello():
  uri = "ws://72.74.137.57:5000"
  count = 0
  start = time.time()
  async with websockets.connect(uri) as ws:
    while True:
      success, frame = cap.read()
      _, buf = cv2.imencode('.jpg', frame)
      jpg_as_text = base64.b64encode(buf)
      await ws.send(jpg_as_text)

      jpg_as_text = await ws.recv()
      buf = base64.b64decode(jpg_as_text)
      img = cv2.imdecode(np.fromstring(buf, dtype=np.int8), -1)
      cv2.imshow('frame', img)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

      count += 1
      end = time.time()
      print("FPS", count / (end - start), end="\r")
    
asyncio.get_event_loop().run_until_complete(hello())
