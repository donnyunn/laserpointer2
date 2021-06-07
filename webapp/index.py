from flask import Flask, request, render_template, url_for, redirect
from datetime import datetime
from random import *
import os
import glob
import time
import math
import threading
import projectivegeometry as pg
import pandas as pd
import numpy as np

import arduino as ard
import cv2

app = Flask(__name__)

CAMERA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/billiard_main.py'
RESOURCE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/resources/'
HW_SETUP = 'hw_setup.txt'

X0 = 1420
Y0 = 710

coord = {
    'movex1':0, 'movey1':0,
    'movex2':0, 'movey2':0,
    'movex3':0, 'movey3':0,
    'offsetx1':-30, 'offsety1':210,
    'offsetx2':0, 'offsety2':230,
    'offsetx3':130, 'offsety3':260,
    'laser1':19952,
    'laser2':19952,
    'laser3':19952,
}

started = 0
recordBtnNameList = {
    str(0):'Game Start',
    str(1):'Stop Recording'
}
recordMsgList = {
    str(0):'Stopped.',
    str(1):'Recording..'
}

filepaths = glob.glob(RESOURCE_PATH + '*.csv')
filenames = []
records = {}
fileusing = []

ini_file = os.path.dirname(os.path.abspath(__file__)) + '/ini_data/ini_data.csv'
ini_data = pd.read_csv(ini_file, header = None)
tr = []
tr.append(ini_data[0][0])
tr.append(ini_data[0][3])
bl = []
bl.append(ini_data[0][2])
bl.append(ini_data[0][1])
center = (int((bl[0]+tr[0])/2) , int((bl[1]+tr[1])/2))

correction_flag = False

@app.route('/')
def index():
    return render_template('index.html', \
        _coord=coord, \
        _recordBtnName=recordBtnNameList[str(started)], \
        _recordMsg=recordMsgList[str(started)], \
        _filenames=getFilenames(), \
        _records=records, \
        _fileusing=fileusing \
        )

@app.route('/hwreset')
def hwreset():
    global correction_flag
    correction_flag = False
    ard.power_reset()
    time.sleep(1)
    # correction_flag = True
    initialization()
    return redirect(url_for('index'))

@app.route('/laser/<int:data>')
def laser(data):
    if data == 0:
        ard.laser_off(0)
        ard.laser_off(1)
        ard.laser_off(2)
    elif data == 1:
        ard.laser_on(0)
        ard.laser_on(1)
        ard.laser_on(2)
    return redirect(url_for('index'))

@app.route('/random')
def random():
    coord['laser1'] = round(randint(1, 39903), -1)
    coord['laser2'] = round(randint(1, 39903), -1)
    coord['laser3'] = round(randint(1, 39903), -1)
    coord['movex1'] = 10 * ((coord['laser1']-1) % (X0*2/10-1) + 1 - (X0/10))
    coord['movey1'] = (Y0 - 10) - 10*((coord['laser1']-1) // (X0*2/10-1))
    coord['movex2'] = 10 * ((coord['laser2']-1) % (X0*2/10-1) + 1 - (X0/10))
    coord['movey2'] = (Y0 - 10) - 10*((coord['laser2']-1) // (X0*2/10-1))
    coord['movex3'] = 10 * ((coord['laser3']-1) % (X0*2/10-1) + 1 - (X0/10))
    coord['movey3'] = (Y0 - 10) - 10*((coord['laser3']-1) // (X0*2/10-1))

    return redirect(url_for('index'))

@app.route('/move', methods = ['POST'])
def move():
    global correction_flag
    if request.method == 'POST':
        coord['laser1'] = int(request.form['laser1'])
        coord['movex1'] = int(float(request.form['moveX1']) * 10)
        coord['movey1'] = int(float(request.form['moveY1']) * 10)
        # x, y = transCoord1(coord['movex1'], coord['movey1'])
        x, y = coord['movex1'], coord['movey1']
        ard.laser_move(0, x, y)
        laser0 = (x, y)
        
        coord['laser2'] = int(request.form['laser2'])
        coord['movex2'] = int(float(request.form['moveX2']) * 10)
        coord['movey2'] = int(float(request.form['moveY2']) * 10)
        # x, y = transCoord2(coord['movex2'], coord['movey2'])
        x, y = coord['movex2'], coord['movey2']
        ard.laser_move(1, x, y)
        laser1 = (x, y)
        
        coord['laser3'] = int(request.form['laser3'])
        coord['movex3'] = int(float(request.form['moveX3']) * 10)
        coord['movey3'] = int(float(request.form['moveY3']) * 10)
        # x, y = transCoord3(coord['movex3'], coord['movey3'])
        x, y = coord['movex3'], coord['movey3']
        ard.laser_move(2, x, y)
        laser2 = (x, y)

        correction_flag = True
        time.sleep(2)
        correct((laser0, laser1, laser2))

        ard.laser_on(0)
        ard.laser_on(1)
        ard.laser_on(2)
    
    return redirect(url_for('index'))

@app.route('/start')
def start():
    global started
    if os.path.isfile(RESOURCE_PATH + str(started)):
        os.remove(RESOURCE_PATH + str(started))

    if started is 0:

        t = threading.Thread(target=threadCamera)
        t.start()

        started = 1
        now = datetime.now().strftime("%y%m%d%H%M.csv")
        f = open(RESOURCE_PATH + now, 'w')
        f.close()

        f = open(RESOURCE_PATH + str(started), 'w')
        f.write(now)
        f.close()

    elif started is 1:
        started = 0
        f = open(RESOURCE_PATH + str(started), 'w')
        f.close()

    return redirect(url_for('index'))

@app.route('/readfile', methods = ['POST'])
def readfile():
    if request.method == 'POST':
        filename = request.form['filenames']
        filepath = RESOURCE_PATH+filename[2:4]+filename[5:7]+filename[8:10]+filename[11:13]+filename[14:16]+'.csv'
        f = open(filepath, 'r')
        lines = f.readlines()
        f.close()
        
        records.clear()
        for line in lines:
            key = line.split(',')[0]
            records[key] = line.split(',')[1:7]
            
            # for 1-dimesional representation
            records[key].append(str(int(((int(round(float(records[key][0])))*10 + X0) + ((0.2*X0-1) * ((Y0-10) - int(round(float(records[key][1])))*10)))/10)))
            records[key].append(str(int(((int(round(float(records[key][2])))*10 + X0) + ((0.2*X0-1) * ((Y0-10) - int(round(float(records[key][3])))*10)))/10)))
            records[key].append(str(int(((int(round(float(records[key][4])))*10 + X0) + ((0.2*X0-1) * ((Y0-10) - int(round(float(records[key][5])))*10)))/10)))
        
        fileusing.clear()
        fileusing.append(filename)

    return redirect(url_for('index'))

@app.route('/update/<key>')
def update(key):
    coord['movex1'] = int(float(records[str(key)][0])*10)
    coord['movey1'] = int(float(records[str(key)][1])*10)
    coord['movex2'] = int(float(records[str(key)][2])*10)
    coord['movey2'] = int(float(records[str(key)][3])*10)
    coord['movex3'] = int(float(records[str(key)][4])*10)
    coord['movey3'] = int(float(records[str(key)][5])*10)
    coord['laser1'] = int(records[str(key)][6])
    coord['laser2'] = int(records[str(key)][7])
    coord['laser3'] = int(records[str(key)][8])
    return redirect(url_for('index'))

@app.route('/deletefile')
def deletefile():
    os.system('rm -f ' + RESOURCE_PATH + '*.csv')
    return redirect(url_for('index'))

@app.route('/poweroff/<int:data>')
def poweroff(data):
    ard.power_off()
    if data == 0:
        t = threading.Thread(target=threadPoweroff)
        t.start()
    elif data == 1:
        os.system('reboot')
    return redirect(url_for('index'))

def transCoord1(x_camera, y_camera):
    x_laser, y_laser = white2laser1(x_camera/10, y_camera/10)
    print(int(round(float(x_laser))), int(round(float(y_laser))))
    return int(round(float(x_laser))), int(round(float(y_laser)))

def white2laser1(x_camera, y_camera):
    if (y_camera >= 35.0):
        if (x_camera < -70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 1)
        elif (x_camera < -35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 2)
        elif (x_camera < 0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 3)
        elif (x_camera < 35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 4)
        elif (x_camera < 70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 5)
        else:
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 6)
    elif (y_camera >= 0):
        if (x_camera < -70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 8)
        elif (x_camera < -35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 9)
        elif (x_camera < 0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 10)
        elif (x_camera < 35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 11)
        elif (x_camera < 70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 12)
        else:
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 13)
    elif (y_camera >= -35.0):
        if (x_camera < -70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 15)
        elif (x_camera < -35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 16)
        elif (x_camera < 0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 17)
        elif (x_camera < 35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 18)
        elif (x_camera < 70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 19)
        else:
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 20)
    else:
        if (x_camera < -70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 22)
        elif (x_camera < -35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 23)
        elif (x_camera < 0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 24)
        elif (x_camera < 35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 25)
        elif (x_camera < 70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 26)
        else:
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 27)
    x_laser = ((A*x_camera) + (B*y_camera) + C) / ((G*x_camera) + (H*y_camera) + 1)
    y_laser = ((D*x_camera) + (E*y_camera) + F) / ((G*x_camera) + (H*y_camera) + 1)
    
    return x_laser*10, y_laser*10

def transCoord2(x_camera, y_camera):
    x_laser, y_laser = yellow2laser2(x_camera/10, y_camera/10)
    print(int(round(float(x_laser))), int(round(float(y_laser))))
    return int(round(float(x_laser))), int(round(float(y_laser)))

def yellow2laser2(x_camera, y_camera):
    if (y_camera >= 35.0):
        if (x_camera < -70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 1)
        elif (x_camera < -35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 2)
        elif (x_camera < 0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 3)
        elif (x_camera < 35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 4)
        elif (x_camera < 70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 5)
        else:
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 6)
    elif (y_camera >= 0):
        if (x_camera < -70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 8)
        elif (x_camera < -35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 9)
        elif (x_camera < 0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 10)
        elif (x_camera < 35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 11)
        elif (x_camera < 70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 12)
        else:
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 13)
    elif (y_camera >= -35.0):
        if (x_camera < -70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 15)
        elif (x_camera < -35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 16)
        elif (x_camera < 0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 17)
        elif (x_camera < 35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 18)
        elif (x_camera < 70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 19)
        else:
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 20)
    else:
        if (x_camera < -70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 22)
        elif (x_camera < -35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 23)
        elif (x_camera < 0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 24)
        elif (x_camera < 35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 25)
        elif (x_camera < 70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 26)
        else:
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 27)
    x_laser = ((A*x_camera) + (B*y_camera) + C) / ((G*x_camera) + (H*y_camera) + 1)
    y_laser = ((D*x_camera) + (E*y_camera) + F) / ((G*x_camera) + (H*y_camera) + 1)
    
    return x_laser*10, y_laser*10

def transCoord3(x_camera, y_camera):
    x_laser, y_laser = red2laser3(x_camera/10, y_camera/10)
    print(int(round(float(x_laser))), int(round(float(y_laser))))
    return int(round(float(x_laser))), int(round(float(y_laser)))

def red2laser3(x_camera, y_camera):
    if (y_camera >= 35.0):
        if (x_camera < -70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 1)
        elif (x_camera < -35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 2)
        elif (x_camera < 0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 3)
        elif (x_camera < 35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 4)
        elif (x_camera < 70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 5)
        else:
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 6)
    elif (y_camera >= 0):
        if (x_camera < -70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 8)
        elif (x_camera < -35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 9)
        elif (x_camera < 0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 10)
        elif (x_camera < 35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 11)
        elif (x_camera < 70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 12)
        else:
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 13)
    elif (y_camera >= -35.0):
        if (x_camera < -70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 15)
        elif (x_camera < -35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 16)
        elif (x_camera < 0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 17)
        elif (x_camera < 35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 18)
        elif (x_camera < 70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 19)
        else:
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 20)
    else:
        if (x_camera < -70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 22)
        elif (x_camera < -35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 23)
        elif (x_camera < 0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 24)
        elif (x_camera < 35.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 25)
        elif (x_camera < 70.0):
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 26)
        else:
            A, B, C, D, E, F, G, H = pg.getMatrix2(0, 27)
    x_laser = ((A*x_camera) + (B*y_camera) + C) / ((G*x_camera) + (H*y_camera) + 1)
    y_laser = ((D*x_camera) + (E*y_camera) + F) / ((G*x_camera) + (H*y_camera) + 1)
    
    return x_laser*10, y_laser*10

def initialization():
    global correction_flag
    ard.power_on()
    f = open(RESOURCE_PATH + HW_SETUP, 'r')
    lines = f.readlines()
    f.close()

    addrs = lines[0].split(',')
    initdatum = []
    initdatum.append(lines[1].split(','))
    initdatum.append(lines[2].split(','))
    initdatum.append(lines[3].split(','))
    ard.init_setup(addrs, initdatum)

    ard.motor_on(0)
    ard.motor_on(1)
    ard.motor_on(2)

    ard.laser_move(0, 0, 0)
    ard.laser_move(1, 0, 0)
    ard.laser_move(2, 0, 0)

    ard.laser_offset(0, coord['offsetx1'], coord['offsety1'])
    ard.laser_offset(1, coord['offsetx2'], coord['offsety2'])
    ard.laser_offset(2, coord['offsetx3'], coord['offsety3'])

    # time.sleep(1)
    # correction_flag = True
    # correct(((0,0),(0,0),(0,0)))

    t = threading.Thread(target=threadMotorLaserOff)
    t.start()

def getFilenames():
    filepaths = glob.glob(RESOURCE_PATH+'*.csv')
    filenames.clear()
    for path in filepaths:
        name = os.path.basename(path)
        filenames.append('20'+name[0:2]+'-'+name[2:4]+'-'+name[4:6]+' '+name[6:8]+':'+name[8:10])
    filenames.sort(reverse=True)
    return filenames

def tfCoord(xy, topright, botleft, origin):
    x = xy[0]
    y = xy[1]
    xr = (x - origin[0]) * 2840 / (topright[0] - botleft[0])
    yr = (y - origin[1]) * 1420 / (topright[1] - botleft[1])
    return (round(xr, 1), round(yr, 1))
def itfCoord(xy, topright, botleft, origin):
    XRATIO = 0.98
    YRATIO = 0.99
    xr = xy[0]
    yr = xy[1]
    x = xr * (topright[0] - botleft[0]) / 2840 + origin[0]
    y = yr * (topright[1] - botleft[1]) / 1420 + origin[1]
    return (int(round((x-origin[0])*XRATIO+origin[0])), int(round((y-origin[1])*YRATIO+origin[1])))

def threadCamera():
    os.system('python3 ' + CAMERA_PATH)
    print('thread stop')

def threadPoweroff():
    os.system('shutdown -h now')

def threadMotorLaserOff():
    time.sleep(0.1)

    ard.motor_off(0)
    ard.motor_off(1)
    ard.motor_off(2)
    
    ard.laser_off(0)
    ard.laser_off(1)
    ard.laser_off(2)
    print("off")

def recognize_laser(n, vs, target):
    global frame
    result = (0, 0)
    lower_laser = np.array([35, 1, 64])
    upper_laser = np.array([90, 255, 255])

    ard.laser_on(n)
    time.sleep(0.1)    
    ret, frame = vs.read()
    if not ret:
        print("camera error")
        return ret, result
    
    ret, frame = vs.read()
    if not ret:
        print("camera error")
        return ret, result
    
    ret, frame = vs.read()
    if not ret:
        print("camera error")
        return ret, result
    
    ret, frame = vs.read()
    if not ret:
        print("camera error")
        return ret, result
        
    frameROI = frame[tr[1]:bl[1], bl[0]:tr[0]]
    hsv = cv2.cvtColor(frameROI, cv2.COLOR_BGR2HSV)
    laser_range = cv2.inRange(hsv, lower_laser, upper_laser)
    laser = cv2.bitwise_and(frameROI, frameROI, mask=laser_range)
    ret, thrlaser = cv2.threshold(laser, 1, 255, cv2.THRESH_BINARY)
    rgblaser = cv2.cvtColor(thrlaser, cv2.COLOR_HSV2RGB)
    graylaser = cv2.cvtColor(rgblaser, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(graylaser, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ret = False
    for cont in contours:
        approx = cv2.approxPolyDP(cont, cv2.arcLength(cont, True) * 0.02, True)
        if len(approx) > 4:
            area = cv2.contourArea(cont)
            if area != 0:
                _, radius = cv2.minEnclosingCircle(cont)
                # if radius > 5 and radius < 30:
                if radius > 1.8 and radius < 5:
                    ratio = radius * radius * math.pi / area
                    print("radius(%f)"%radius)
                    print("ratio(%f)"%ratio)
                    if ratio < 9:
                        (x, y, w, h) = cv2.boundingRect(cont)
                        # result = (int((x+x+w)/2), int((y+y+h)/2))
                        # if result[0] > bl[0]-10 and result[0] < tr[0]+10 and result[1] > tr[1]-10 and result[1] < bl[1]+10:
                        if True:
                            # print("radius(%f)"%radius)
                            # print("ratio(%f)"%ratio)
                            result = (int((x+x+w)/2)+bl[0], int((y+y+h)/2)+tr[1])
                            ret = True

                            pt1 = (x-10, y-10)
                            pt2 = (x + w+10, y + h + 10)
                            cv2.rectangle(laser, pt1, pt2, (0, 0, 255), 1)

                            break
                        else:
                            cv2.circle(laser, result, 5, (0, 0, 255), -1)

    # pt1 = (tr[0], tr[1])
    # pt2 = (bl[0], bl[1])
    pt1 = (0, tr[0]-bl[0])
    pt2 = (0, bl[1]-tr[1])
    cv2.rectangle(laser, pt1, pt2, (0, 255, 0), 2)
    cv2.circle(laser, (center[0]-bl[0], center[1]-tr[1]), 1, (255, 255, 255), -1)
    cv2.circle(laser, (result[0]-bl[0], result[1]-tr[1]), 1, (0, 0, 255), -1)
    cv2.circle(laser, (target[0]-bl[0], target[1]-tr[1]), 3, (0, 255, 255), -1)
    # cv2.imshow("Frame", laser)

    # ret_, frame = vs.read()
    # if not ret_:
    #     return ret_, result

    return ret, result

def correct(laser_req):
    global correction_flag
    if correction_flag is False:
        return
    # print(laser_req)
    # return
    ard.laser_off(0)
    ard.laser_off(1)
    ard.laser_off(2)
    vs = cv2.VideoCapture(-1)
    vs.set(3, 1920)
    vs.set(4, 1080)
    laser = (0, 0)

    for n in range(3):
        ard.motor_on(n)
        time.sleep(0.5)
        retry = 0
        xtick = 1
        ytick = 1
        target = itfCoord(laser_req[n], tr, bl, center)
        while correction_flag:            
            ret, laser = recognize_laser(n, vs, target)
            if ret:
                retry = retry + 1
                x_err = int(target[0] - laser[0])
                y_err = int(target[1] - laser[1])
                print(target)
                print(laser)
                print(x_err, y_err)

                if abs(x_err) > 150 or abs(y_err) > 150:
                    print("too far")
                    laser = (0, 0)
                    retry = retry + 1
                else:
                    if abs(x_err) < 3 and abs(y_err) < 3:
                        print("corrected (%d)"%n)
                        break;
                    else:
                        if abs(x_err) >= 10:
                            if x_err < 0:
                                xtick = -3
                            else:
                                xtick = 3
                        elif abs(x_err) >= 3:
                            if x_err < 0:
                                xtick = -1
                            else:
                                xtick = 1
                        else:
                            xtick = 0
                        if abs(y_err) >= 10:
                            if y_err < 0:
                                ytick = 3
                            else:
                                ytick = -3
                        elif abs(y_err) >= 3:
                            if y_err < 0:
                                ytick = 1
                            else:
                                ytick = -1
                        else:
                            ytick = 0
                    ard.laser_tick(n, xtick, ytick)

            else:
                laser = (0, 0)
                retry = retry + 1
                # x, y = (0, 0)
                print(".")
            if retry > 15:
                print("fail(%d)"%n)
                retry = 0
                break;
            if cv2.waitKey(1) == 27:
                break
        ard.laser_off(n)

    # while True:
    #     # ret, frame = vs.read()
    #     # if not ret:
    #     #     print(ret)
    #     #     break
    #     ret, laser = recognize_laser(0, vs)
    #     if ret:
    #         x, y = tfCoord(laser, tr, bl, center)
    #         print(x, y)
    #     # cv2.imshow("Frame", frame)
    #     if cv2.waitKey(1) == 27:
    #         ard.laser_tick(0, -int(x/10), -int(y/10))
    #         break;
    
    # ard.motor_off(0)
    # ard.motor_off(1)
    # ard.motor_off(2)
    ard.laser_off(0)
    ard.laser_off(1)
    ard.laser_off(2)

    correction_flag = False

    vs.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    correction_flag = False
    ard.power_off()
    time.sleep(1)
    ard.power_on()
    time.sleep(1)
    initialization()
    app.run(debug=True, port=80, host='0.0.0.0')
