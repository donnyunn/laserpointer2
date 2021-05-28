import os, math
import cv2
import numpy as np
import pandas as pd
import recogplay as rp

X0 = 284.0
Y0 = 142.0


def setLabel(img, pts):
    (x, y, w, h) = cv2.boundingRect(pts)
    # pt1 = (x, y)
    # pt2 = (x + w, y + h)
    # cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
    return (int((x+x+w)/2), int((y+y+h)/2))

def setTableLine(img, tr, bl, center):
    pt1 = (tr[0], tr[1])
    pt2 = (bl[0], bl[1])
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.circle(img, center, 3, (255, 0, 255), -1)
    
def setPoint(frame, point):
    cv2.circle(frame, point, 3, (255, 128, 64), -1)

def tfCoord(xy, topright, botleft, origin):
    x = xy[0]
    y = xy[1]
    xr = (x - origin[0]) * X0 / (topright[0] - botleft[0])
    yr = (y - origin[1]) * Y0 / (topright[1] - botleft[1])
    return (round(xr, 1), round(yr, 1))

def recognize_white(frame, tr, bl):
    result = (0, 0)
    # frame_ROI = frame[tr[1]:bl[1], bl[0]:tr[0]]
    frame_ROI = frame

    lower_white = np.array([0, 0, 128])
    upper_white = np.array([180, 64, 255])

    hsv = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2HSV)
    white_range = cv2.inRange(hsv, lower_white, upper_white)
    white = cv2.bitwise_and(frame_ROI, frame_ROI, mask=white_range)

    ret, thrwhite = cv2.threshold(white, 1, 255, cv2.THRESH_BINARY)
    rgbwhite = cv2.cvtColor(thrwhite, cv2.COLOR_HSV2RGB)
    graywhite = cv2.cvtColor(rgbwhite, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(graywhite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cont in contours:
        approx = cv2.approxPolyDP(cont, cv2.arcLength(cont, True) * 0.02, True)
        if len(approx) > 5:
            area = cv2.contourArea(cont)
            if area != 0:
                _, radius = cv2.minEnclosingCircle(cont)
                if radius > 15 and radius < 25:
                    ratio = radius * radius * math.pi / area
                    if int(ratio) == 1:
                        result = setLabel(frame, cont)
        
    return result

def recognize_yellow(frame, tr, bl):
    result = (0, 0)
    # frame_ROI = frame[tr[1]:bl[1], bl[0]:tr[0]]
    frame_ROI = frame

    lower_yellow = np.array([15, 128, 128])
    upper_yellow = np.array([45, 255, 255])

    hsv = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2HSV)
    yellow_range = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow = cv2.bitwise_and(frame_ROI, frame_ROI, mask=yellow_range)

    ret, thryellow = cv2.threshold(yellow, 1, 255, cv2.THRESH_BINARY)
    rgbyellow = cv2.cvtColor(thryellow, cv2.COLOR_HSV2RGB)
    grayyellow = cv2.cvtColor(rgbyellow, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(grayyellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cont in contours:
        approx = cv2.approxPolyDP(cont, cv2.arcLength(cont, True) * 0.02, True)
        if len(approx) > 5:
            area = cv2.contourArea(cont)
            if area != 0:
                _, radius = cv2.minEnclosingCircle(cont)
                if radius > 15 and radius < 25:
                    ratio = radius * radius * math.pi / area
                    if int(ratio) == 1:
                        result = setLabel(frame, cont)
    
    return result

def recognize_red(frame, tr, bl):
    result = (0, 0)
    # frame_ROI = frame[tr[1]:bl[1], bl[0]:tr[0]]
    frame_ROI = frame

    lower_red = np.array([165, 128, 128])
    upper_red = np.array([195, 255, 255])

    hsv = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2HSV)
    red_range = cv2.inRange(hsv, lower_red, upper_red)
    red = cv2.bitwise_and(frame_ROI, frame_ROI, mask=red_range)

    ret, thrred = cv2.threshold(red, 1, 255, cv2.THRESH_BINARY)
    rgbred = cv2.cvtColor(thrred, cv2.COLOR_HSV2RGB)
    grayred = cv2.cvtColor(rgbred, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(grayred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cont in contours:
        approx = cv2.approxPolyDP(cont, cv2.arcLength(cont, True) * 0.02, True)
        if len(approx) > 5:
            area = cv2.contourArea(cont)
            if area != 0:
                _, radius = cv2.minEnclosingCircle(cont)
                if radius > 15 and radius < 25:
                    ratio = radius * radius * math.pi / area
                    if int(ratio) == 1:
                        result = setLabel(frame, cont)
    
    return result

def loop():
    file_1_directory = os.path.dirname(os.path.abspath(__file__)) + '/resources/1'
    ini_file = os.path.dirname(os.path.abspath(__file__)) + '/ini_data/ini_data.csv'
    f = open(file_1_directory, 'r')
    save_file = f.read()
    save_file = os.path.dirname(os.path.abspath(__file__)) + '/resources/' + save_file
    f.close()

    ini_data = pd.read_csv(ini_file, header = None)
    tr = []
    tr.append(ini_data[0][0])
    tr.append(ini_data[0][3])
    bl = []
    bl.append(ini_data[0][2])
    bl.append(ini_data[0][1])
    center = (int((bl[0]+tr[0])/2) , int((bl[1]+tr[1])/2))
    print(bl, tr, center)

    vs = cv2.VideoCapture(-1)
    vs.set(3, 1920)
    vs.set(4, 1080)

    while os.path.isfile(file_1_directory):
        ret, frame = vs.read()
        if not ret:
            print("cam error")
            continue
        
        whitepoint = recognize_white(frame, tr, bl)
        yellowpoint = recognize_yellow(frame, tr, bl)
        redpoint = recognize_red(frame, tr, bl)
        
        if whitepoint != (0, 0) and yellowpoint != (0, 0) and redpoint != (0, 0):
            whitepoint_real = tfCoord(whitepoint, tr, bl, center)
            yellowpoint_real = tfCoord(yellowpoint, tr, bl, center)
            redpoint_real = tfCoord(redpoint, tr, bl, center)

            rp.billiardRule(save_file, whitepoint_real, yellowpoint_real, redpoint_real)

        # setTableLine(frame, tr, bl, center)
        # setPoint(frame, whitepoint)
        # setPoint(frame, yellowpoint)
        # setPoint(frame, redpoint)
        # cv2.imshow('frame', frame)

        # if cv2.waitKey(1) == 27:
        #     break
        # time.sleep(1)
    
    vs.release()
    cv2.destroyAllWindows()
