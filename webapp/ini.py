import cv2
import numpy as np
import os

ini_file = os.path.dirname(os.path.abspath(__file__)) + '/ini_data/ini_data.csv'

lower_blue = np.array([75, 128, 96])
upper_blue = np.array([120, 255, 255])

vs = cv2.VideoCapture(-1)
vs.set(3, 1920)
vs.set(4, 1080)

for i in range(10):
    ret, frame = vs.read()
    repeat = 0
    while ret is False:
        repeat = repeat + 1
        ret, frame = vs.read()
        if repeat > 20:
            break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_range = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=blue_range)

    ret, threshold = cv2.threshold(res, 1, 255, cv2.THRESH_BINARY)
    rgbtable = cv2.cvtColor(threshold, cv2.COLOR_HSV2RGB)
    graytable = cv2.cvtColor(rgbtable, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(graytable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cont in contours:
        approx = cv2.approxPolyDP(cont, cv2.arcLength(cont, True) * 0.02, True)
        vtc = len(approx)
        if vtc == 4:
            (x, y, w, h) = cv2.boundingRect(cont)
            if w > 100 or h > 100:
                pt1 = (x+30, y+30)
                pt2 = (x+w-30, y+h-30)

f = open(ini_file, 'w')
f.write(str(pt2[0])+'\n')
f.write(str(pt2[1])+'\n')
f.write(str(pt1[0])+'\n')
f.write(str(pt1[1])+'\n')
f.close()
