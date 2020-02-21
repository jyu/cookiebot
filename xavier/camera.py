import cv2
import subprocess

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L)
_, img = cap.read()
cv2.imwrite("img.jpg", img)
