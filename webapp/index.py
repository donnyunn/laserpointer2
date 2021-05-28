from flask import Flask, request, render_template, url_for, redirect
from datetime import datetime
from random import *
import os
import glob
import time
import threading

import arduino as ard

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
    'offsetx1':-170, 'offsety1':130,
    'offsetx2':-50, 'offsety2':40,
    'offsetx3':360, 'offsety3':430,
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
    ard.power_reset()
    time.sleep(1)
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
    if request.method == 'POST':
        coord['laser1'] = int(request.form['laser1'])
        coord['movex1'] = int(float(request.form['moveX1']) * 10)
        coord['movey1'] = int(float(request.form['moveY1']) * 10)
        ard.laser_move(0, coord['movex1'], coord['movey1'])
        
        coord['laser2'] = int(request.form['laser2'])
        coord['movex2'] = int(float(request.form['moveX2']) * 10)
        coord['movey2'] = int(float(request.form['moveY2']) * 10)
        ard.laser_move(1, coord['movex2'], coord['movey2'])
        
        coord['laser3'] = int(request.form['laser3'])
        coord['movex3'] = int(float(request.form['moveX3']) * 10)
        coord['movey3'] = int(float(request.form['moveY3']) * 10)
        ard.laser_move(2, coord['movex3'], coord['movey3'])

        correct()
    
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

def initialization():
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

    ard.laser_offset(0, coord['offsetx1'], coord['offsety1'])
    ard.laser_offset(1, coord['offsetx2'], coord['offsety2'])
    ard.laser_offset(2, coord['offsetx3'], coord['offsety3'])

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

def threadCamera():
    os.system('python3 ' + CAMERA_PATH)
    print('thread stop')

def threadPoweroff():
    os.system('shutdown -h now')

def threadMotorLaserOff():
    time.sleep(4)

    ard.motor_off(0)
    ard.motor_off(1)
    ard.motor_off(2)
    
    ard.laser_off(0)
    ard.laser_off(1)
    ard.laser_off(2)

def correct():
    time.sleep(2)
    ard.motor_off(0)
    ard.motor_off(1)
    ard.motor_off(2)

if __name__ == '__main__':
    ard.power_off()
    time.sleep(1)
    ard.power_on()
    time.sleep(1)
    initialization()
    app.run(debug=True, port=80, host='0.0.0.0')
