from collections import deque, Counter
import numpy as np

import cv2
import imutils

import time
import pandas as pd
import math
import os

def variables_setting():
    global folder_directory, file_1, file_1_directory, file_1_name
    folder_directory = os.path.dirname(os.path.abspath(__file__)) + '/ini_data/'
    video_path = os.path.dirname(os.path.abspath(__file__)) + '/billiards2.mp4'
    video_path = -1


    global low_white, high_white, low_yellow, high_yellow, low_red, high_red, low_table, high_table, vs, frame, ret, min_ball_size
    min_ball_size = 18
    low_white = np.array([0, 0, 120], dtype=np.uint8)
    high_white = np.array([255, 100, 255], dtype=np.uint8)

    low_yellow = np.array([21, 97, 10], dtype=np.uint8)
    high_yellow = np.array([50, 255, 255], dtype=np.uint8)

    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])

    low_table = np.array([0, 0, 90], dtype=np.uint8)
    high_table = np.array([255, 255, 255], dtype=np.uint8)

    vs = cv2.VideoCapture(video_path)
    ret, frame = vs.read()
    vs.set(3, 1920)
    vs.set(4, 1080)

    global t0, current_t, rest_time, initialization_time, noise_time, FPS_INTERVAL, frame_count, fps_time, tl, tr, bl, br
    global a, a_r, result_array, b, b2, b3, b4, b_r, b2_r, b3_r, b4_r
    # Time
    t0 = time.time()
    current_t = round(time.time() - t0, 2)
    rest_time = 1
    initialization_time = 5
    noise_time = 3

    FPS_INTERVAL = 10  # updates FPS estimate after this many frames
    frame_count = 0
    fps_time = time.time()

    tl = np.zeros(2)
    tr = np.zeros(2)
    bl = np.zeros(2)
    br = np.zeros(2)

    a = np.zeros(12)
    a_r = np.zeros(12)
    result_array = np.zeros(7)

    b = [0, 0, 0]
    b2 = [0, 0, 0]
    b3 = [0, 0, 0]
    b4 = [0, 0, 0]

    b_r = [0, 0, 0]
    b2_r = [0, 0, 0]
    b3_r = [0, 0, 0]
    b4_r = [0, 0, 0]

    global game_num, game_phase, switch_shot, switch_end, time_shot, time_shot_length, time_end, time_end_length, shot_loss_length, shot_rate, end_loss_length, end_rate
    global noise_loss_length, data_length
    game_num = 0
    game_phase = 1  # 1 : shot check, 0 : end check
    switch_shot = 0
    switch_end = 0
    time_shot = 0
    time_shot_length = 6
    time_end = 0
    time_end_length = 0.1

    noise_loss_length = 20

    shot_loss_length = 5
    shot_rate = 10

    end_loss_length = 5
    end_rate = 2

    data_length = 300
    global stack, raw, kkk, table_real_length, table_real_height, table_real_bumper, key_value
    stack = [0, 0, 0, 0, 0, 0, 0]
    raw = [0, 0, 0, 0, 0, 0, 0]
    kkk = 1.05

    table_real_length = 2945
    table_real_height = 1521
    table_real_bumper = 32  # 52
    key_value = 0

    global hand_filter_time, time_count
    hand_filter_time = 0
    time_count = 0

def real_coordinate(value, center, max, min, real_length):
    return (value - center) * real_length / (max - min)
    
def initialize_table_recog():
    global t0, initialization_time, current_t
    global vs, frame, ret, FPS_INTERVAL, frame_count, fps_time
    global low_table, high_table, tl, tr, bl, br
    global radius_table, x_table, y_table
    global table_real_length, table_real_height, table_real_bumper
    global tr, bl
    table_list = np.zeros(3)

    while current_t <= initialization_time / 2:
        current_t = round(time.time() - t0, 2)
        ret, frame = vs.read()

        if not (frame_count % 10):
            fps = FPS_INTERVAL / (time.time() - fps_time)
            fps_time += 1

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_table = cv2.inRange(hsv, low_table, high_table)
        mask_table = cv2.erode(mask_table, None, iterations=2)
        mask_table = cv2.dilate(mask_table, None, iterations=2)

        cnts_table = cv2.findContours(mask_table.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        cnts_table = imutils.grab_contours(cnts_table)
        center_table = None

        if len(cnts_table) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c_table = max(cnts_table, key=cv2.contourArea)
            ((x_table, y_table), radius_table) = cv2.minEnclosingCircle(c_table)
            M_table = cv2.moments(c_table)
            center_table = (int(M_table["m10"] / M_table["m00"]), int(M_table["m01"] / M_table["m00"]))

            if radius_table > 5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x_table), int(y_table)), int(radius_table),
                           (0, 255, 255), 2)
                cv2.circle(frame, center_table, 5, (0, 0, 255), -1)
                table_new = np.array([round(radius_table, 2), round(x_table, 2), round(y_table, 2)])
                table_list = np.vstack([table_list, table_new])

    table_list_radius = table_list[:, 0]
    counter = Counter(table_list_radius)
    most_freq = counter.most_common(1)
    most_freq = np.array(most_freq)
    print("table_label value : ", most_freq[0][0])  # most common value
    print("table_label No. : ", most_freq[0][1])  # most common number

    most_freq_counter = int(most_freq[0][1])

    radius_table, x_table, y_table = table_list[most_freq_counter]
    print("Table properties : ", radius_table, x_table, y_table)

    table_ratio = table_real_height / table_real_length
    # a, b : table outer radius
    # aa, bb : table inner radius_bumber
    table_a = radius_table / (math.sqrt(1 + table_ratio ** 2))
    table_aa = table_a * (table_real_length / 2 - table_real_bumper) / (table_real_length / 2)
    table_b = table_ratio * table_a
    table_bb = table_b * (table_real_height / 2 - table_real_bumper) / (table_real_height / 2)
    tl = (x_table - table_aa, y_table + table_bb)
    tr = (x_table + table_aa, y_table + table_bb)
    bl = (x_table - table_aa, y_table - table_bb)
    br = (x_table + table_aa, y_table - table_bb)
    tl = [int(tl[0]), int(tl[1])]
    tr = [int(tr[0]), int(tr[1])]
    bl = [int(bl[0]), int(bl[1])]
    br = [int(br[0]), int(br[1])]
    print('tl :', tl)
    print('bl :', bl)
    print('tr :', tr)
    print('br :', br)
    frame = cv2.line(frame, (tl[0], tl[1]), (tr[0], tr[1]), (0, 255, 0), 2)
    frame = cv2.line(frame, (tr[0], tr[1]), (br[0], br[1]), (0, 255, 0), 2)
    frame = cv2.line(frame, (br[0], br[1]), (bl[0], bl[1]), (0, 255, 0), 2)
    frame = cv2.line(frame, (bl[0], bl[1]), (tl[0], tl[1]), (0, 255, 0), 2)
    result_array = np.vstack([tr[0], tr[1], bl[0], bl[1]])
    df = pd.DataFrame(result_array)
    df.to_csv(folder_directory + 'ini_data.csv', header=False, index=False)

global folder_directory
key_coin=0
variables_setting()
initialize_table_recog()
while key_coin==0:
    time.sleep(1)
    try :
        variables_setting()
        initialize_table_recog()    
        key_coin=1
    except:
        print("initialzation fail!")
data = pd.read_csv(folder_directory + 'ini_data.csv', header = None)

#print("tr = ", data[0], ", ", data[1])
#print("bl = ", data[2], ", ", data[3])
print("initialzation Success!")
print(data)


    
    
    
