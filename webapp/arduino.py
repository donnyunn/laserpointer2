from gpiozero import LED, exc
import smbus
import time

power = LED(4)
i2c = smbus.SMBus(1)
I2C_INTERVAL = 0.1

ADDR = [int(0x41), int(0x42), int(0x43)]

power_status = False

def power_on():
    global power_status
    power_status = True
    power.off()

def power_off():
    global power_status
    power_status = False
    power.on()

def power_reset():
    power_off()
    time.sleep(0.5)
    power_on()

def laser_on(num):
    tx = [2]
    try:
        time.sleep(I2C_INTERVAL)
        i2c.write_i2c_block_data(ADDR[num], 0x00, tx)
    except:
        print('i2c error(%d): '%num + str(laser_on))

def laser_off(num):
    tx = [1]
    try:
        time.sleep(I2C_INTERVAL)
        i2c.write_i2c_block_data(ADDR[num], 0x00, tx)
    except:
        print('i2c error(%d): '%num + str(laser_off))

def motor_on(num):
    tx = [4]
    try:
        time.sleep(I2C_INTERVAL)
        i2c.write_i2c_block_data(ADDR[num], 0x00, tx)
    except:
        print('i2c error(%d): '%num + str(motor_on))

def motor_off(num):
    tx = [8]
    try:
        time.sleep(I2C_INTERVAL)
        i2c.write_i2c_block_data(ADDR[num], 0x00, tx)
    except:
        print('i2c error(%d): '%num + str(motor_on))

def laser_move(num, x, y):
    tx = []
    tx.append(x>>8 & 0xff)
    tx.append(x & 0xff)
    tx.append(y>>8 & 0xff)
    tx.append(y & 0xff)
    try:
        time.sleep(I2C_INTERVAL)
        i2c.write_i2c_block_data(ADDR[num], 0x01, tx)
    except:
        print('i2c error(%d): '%num + str(laser_move))

def laser_offset(num, x, y):
    tx = []
    tx.append(x>>8 & 0xff)
    tx.append(x & 0xff)
    tx.append(y>>8 & 0xff)
    tx.append(y & 0xff)
    try:
        time.sleep(I2C_INTERVAL)
        i2c.write_i2c_block_data(ADDR[num], 0x05, tx)
    except:
        print('i2c error(%d): '%num + str(laser_move))

def laser_tick(num, x, y):
    tx = []
    tx.append(x & 0xff)
    tx.append(y & 0xff)
    try:
        time.sleep(I2C_INTERVAL)
        i2c.write_i2c_block_data(ADDR[num], 0x0F, tx)
    except:
        print('i2c error(%d): '%num + str(laser_tick))

def init_setup(addrs, datum):
    tx = []
    for i in range(3):
        tx.clear()
        tx.append((int(datum[i][0])*10) >>8 & 0xff)
        tx.append((int(datum[i][0])*10) & 0xff)
        tx.append((int(datum[i][1])*10) >>8 & 0xff)
        tx.append((int(datum[i][1])*10) & 0xff)
        tx.append((int(datum[i][2])*10) >>8 & 0xff)
        tx.append((int(datum[i][2])*10) & 0xff)
        try:
            time.sleep(0.1)
            i2c.write_i2c_block_data(ADDR[i], int(addrs[0].replace("0x",""),16), tx)
        except:
            print('i2c error(%d): '%i + str(init_setup))