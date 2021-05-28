from collections import deque, Counter
import numpy as np

import cv2
import imutils

import time
import pandas as pd
import math
import os


def variables_setting():
    global folder_directory, file_1, file_1_directory, file_1_name, ini_folder_directory, video_path
    folder_directory = os.path.dirname(os.path.abspath(__file__)) + '/resources/'
    ini_folder_directory = os.path.dirname(os.path.abspath(__file__)) + '/ini_data/'
    video_path = os.path.dirname(os.path.abspath(__file__)) + '/billiards2.mp4'
    video_path = -1

    file_1_directory = folder_directory + '1'
    file_1_name = "None"

    file_1 = open(file_1_directory, 'r', )
    file_1_name = file_1.read()
    file_1.close()

    global low_white, high_white, low_yellow, high_yellow, low_red, high_red, low_table, high_table, vs, frame, ret, min_ball_size
    min_ball_size = 12 #5
    low_white = np.array([0, 0, 120], dtype=np.uint8)
    high_white = np.array([255, 100, 255], dtype=np.uint8)

    low_yellow = np.array([21, 20, 10], dtype=np.uint8)
    high_yellow = np.array([70, 255, 255], dtype=np.uint8)

    low_red = np.array([-30, 155, 84])
    high_red = np.array([30, 255, 255])

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
    initialization_time = 3
    noise_time = 3

    FPS_INTERVAL = 10  # updates FPS estimate after this many frames
    frame_count = 0
    fps_time = time.time()

    tl = [0, 0]
    tr = [0, 0]#np.zeros(2)
    bl = [0, 0]
    br = [0, 0]

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

def error_check(past, new, range):
    if abs(new - past) - range > 0:
        value = 0
    else:
        value = 1
    return value

def real_coordinate_x(value, center, max, min, real_length):
     output_val = (value - center) * real_length / (max - min)
     if output_val >= 142:
        output_val = 141.9
     elif output_val <= -142:
        output_val = -141.9
     return (value - center) * real_length / (max - min)
    
def real_coordinate_y(value, center, max, min, real_length):
     output_val = (value - center) * real_length / (max - min)
     if output_val >= 71:
        output_val = 70.9
     elif output_val <= -71:
        output_val = -70.9
     return (value - center) * real_length / (max - min)


def initialize_table_recog():
    global t0, initialization_time, current_t
    global vs, frame, ret, FPS_INTERVAL, frame_count, fps_time
    global low_table, high_table, tl, tr, bl, br
    global radius_table, x_table, y_table
    global table_real_length, table_real_height, table_real_bumper

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

def initialize_table_read():
    global ini_folder_directory, bl, tr
    data = pd.read_csv(ini_folder_directory + 'ini_data.csv', header = None)

    tr[0] = (data[0][0])
    tr[1] = (data[0][1])
    bl[0] = (data[0][2])
    bl[1] = (data[0][3])
    #
    # tr = np.hstack(int(data[0][0]), int(data[0][1]))
    # bl = np.hstack(int(data[0][2]), int(data[0][3]))


def initialize_setting():
    global t0, initialization_time, current_t
    global vs, frame, ret, FPS_INTERVAL, frame_count, fps_time, frame_ROI
    global low_table, high_table, tr, tl, br, bl
    global raw
    global radius_ini, radius2_ini, radius3_ini, min_ball_size
    global kkk
    global video_path
    ball_list_white = np.zeros(3)
    ball_list_yellow = np.zeros(3)
    ball_list_red = np.zeros(3)

    while current_t <= initialization_time:
        current_t = round(time.time() - t0, 2)

        ret = False
        while ret != True:
            ret, frame = vs.read()
        frame_ROI = frame[bl[1]:tr[1], bl[0]:tr[0]]

        if not (frame_count % 10):
            fps = FPS_INTERVAL / (time.time() - fps_time)
            # print('Frame :', frame_count, ' | FPS : ', fps)
            fps_time += 1

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break
        kernel = np.ones((3, 3), np.uint8)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2HSV)

        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, low_white, high_white)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        mask2 = cv2.inRange(hsv, low_yellow, high_yellow)
        mask2 = cv2.erode(mask2, None, iterations=2)
        mask2 = cv2.dilate(mask2, None, iterations=2)

        mask3 = cv2.inRange(hsv, low_red, high_red)
        mask3 = cv2.erode(mask3, None, iterations=2)
        mask3 = cv2.dilate(mask3, None, iterations=2)

        # frame :
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)
        center2 = None

        cnts3 = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cnts3 = imutils.grab_contours(cnts3)
        center3 = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > min_ball_size:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame_ROI, (int(x), int(y)), int(radius),
                           (0, 0, 255), 2)
                cv2.circle(frame_ROI, center, 5, (0, 0, 255), -1)
                b = np.array([radius, x, y], float)
                ball_list_white = np.vstack([ball_list_white, b])

        if len(cnts2) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroidq
            c2 = max(cnts2, key=cv2.contourArea)
            ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
            M2 = cv2.moments(c2)
            center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))

            if radius2 > min_ball_size:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame_ROI, (int(x2), int(y2)), int(radius2),
                           (0, 0, 255), 2)
                cv2.circle(frame_ROI, center2, 5, (255, 255, 255), -1)
                b2 = np.array([radius2, x2, y2])
                ball_list_yellow = np.vstack([ball_list_yellow, b2])

        if len(cnts3) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c3 = max(cnts3, key=cv2.contourArea)
            ((x3, y3), radius3) = cv2.minEnclosingCircle(c3)
            M3 = cv2.moments(c3)
            center3 = (int(M3["m10"] / M3["m00"]), int(M3["m01"] / M3["m00"]))

            if radius3 > min_ball_size:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame_ROI, (int(x3), int(y3)), int(radius3),
                           (0, 255, 255), 2)
                cv2.circle(frame_ROI, center3, 5, (255, 255, 255), -1)
                b3 = np.array([radius3, x3, y3])
                ball_list_red = np.vstack([ball_list_red, b3])

        center4 = (int(((bl[0] + tr[0]) / 2)), int(((bl[1] + tr[1]) / 2)))
        cv2.circle(frame, center4, 5, (0, 0, 255), -1)

    counter_white = Counter(ball_list_white[1:, 1])
    most_freq_white = counter_white.most_common(1)
    most_freq_white = np.array(most_freq_white)
    most_freq_counter_white = int(most_freq_white[0][1])
    radius_ini, raw[1], raw[2] = ball_list_white[most_freq_counter_white - 1, :]

    counter_yellow = Counter(ball_list_yellow[1:, 1])
    most_freq_yellow = counter_white.most_common(1)
    most_freq_yellow = np.array(most_freq_yellow)
    most_freq_counter_yellow = int(most_freq_yellow[0][1])
    radius2_ini, raw[3], raw[4] = ball_list_yellow[most_freq_counter_yellow - 1, :]

    counter_red = Counter(ball_list_red[1:, 1])
    most_freq_red = counter_red.most_common(1)
    most_freq_red = np.array(most_freq_red)
    most_freq_counter_red = int(most_freq_red[0][1])
    radius3_ini, raw[5], raw[6] = ball_list_red[most_freq_counter_red - 1, :]



def recog_balls():
    global a, a_r, b, b2, b3, b4, b_r, b2_r, b3_r, b4_r, bb, bb_r  # pixel level and real coordination level
    global radius_table, x_table, y_table, tl, tr, bl, br
    global current_t, t0
    global vs, ret, frame, frame_ROI, frame_count, FPS_INTERVAL, fps_time
    global circle_shape, table_real_bumper, table_real_length

    current_t = round(time.time() - t0, 2)
    # initialize some stuff
    c_colors = [(0, 0, 255)] * 4

    # grab the current frame
    ret, frame = vs.read()


    # hheight, wwidth, cchannel = frame.shape
    # mmatrix = cv2.getRotationMatrix2D((wwidth, hheight), 0.05, 1)
    # frame = cv2.warpAffine(frame, mmatrix, (wwidth, hheight))

    # kk_1, kk_2, kk_3 = -0.004, -0.001, -0.002
    # rows, cols = frame.shape[:2]
    # mapy, mapx = np.indices((rows, cols), dtype = np.float32)

    # mapx = 2*mapx/(cols-1)-1
    # mapy = 2*mapy/(rows-1)-1
    # rrr, theta = cv2.cartToPolar(mapx, mapy)
    # rrru = rrr*(1+kk_1*(rrr**2) + kk_2*(rrr**4) + kk_3*(rrr**6))
    # mapx, mapy = cv2.polarToCart(rrru, theta)
    # mapx = ((mapx + 1)*cols-1)/2
    # mapy = ((mapy + 1)*rows-1)/2
    # distorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    # frame = distorted
    



    frame_ROI = frame[bl[1]:tr[1], bl[0]:tr[0]]

    # resize the frame, blur it, and convert it to the HSV
    # color space
    kernel = np.ones((3, 3), np.uint8)
    hsv = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2HSV)

    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, low_white, high_white)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    mask2 = cv2.inRange(hsv, low_yellow, high_yellow)
    mask2 = cv2.erode(mask2, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)

    mask3 = cv2.inRange(hsv, low_red, high_red)
    mask3 = cv2.erode(mask3, None, iterations=2)
    mask3 = cv2.dilate(mask3, None, iterations=2)

    # frame :
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    center2 = None

    cnts3 = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    cnts3 = imutils.grab_contours(cnts3)
    center3 = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > min_ball_size and radius < radius_ini * kkk:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame_ROI, (int(x), int(y)), int(radius), (0, 0, 255), 2)  # 2
            cv2.circle(frame_ROI, center, 5, (0, 0, 255), -1)  # -1
        b = np.array([radius, x, y])
        b_r = np.array([(real_coordinate_x(radius, 0, tr[0], bl[0], table_real_length - 2*table_real_bumper) + real_coordinate_y(radius, 0, tr[1],
                                                                                                       bl[1],
                                                                                                       table_real_height - 2*table_real_bumper)) / 2,
                        real_coordinate_x(x, (tr[0] - bl[0]) / 2, tr[0], bl[0], table_real_length - 2*table_real_bumper),
                        real_coordinate_y(y, (tr[1] - bl[1]) / 2, tr[1], bl[1], table_real_height - 2*table_real_bumper)])

    if len(cnts2) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroidq
        c2 = max(cnts2, key=cv2.contourArea)
        ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
        M2 = cv2.moments(c2)
        center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))

        if radius2 > min_ball_size and radius2 < radius2_ini * kkk:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame_ROI, (int(x2), int(y2)), int(radius2), (0, 0, 255), 2)  # 2
            cv2.circle(frame_ROI, center2, 5, (255, 255, 255), -1)  # -1
        b2 = np.array([radius2, x2, y2])
        b2_r = np.array([(real_coordinate_x(radius2, 0, tr[0], bl[0], table_real_length - 2*table_real_bumper) + real_coordinate_y(radius2, 0,
                                                                                                         tr[1], bl[1],
                                                                                                         table_real_height - 2*table_real_bumper)) / 2,
                         real_coordinate_x(x2, (tr[0] - bl[0]) / 2, tr[0], bl[0], table_real_length - 2*table_real_bumper),
                         real_coordinate_y(y2, (tr[1] - bl[1]) / 2, tr[1], bl[1], table_real_height - 2*table_real_bumper)])

    if len(cnts3) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c3 = max(cnts3, key=cv2.contourArea)
        ((x3, y3), radius3) = cv2.minEnclosingCircle(c3)
        M3 = cv2.moments(c3)
        center3 = (int(M3["m10"] / M3["m00"]), int(M3["m01"] / M3["m00"]))

        if radius3 > min_ball_size and radius3 < radius3_ini * kkk:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame_ROI, (int(x3), int(y3)), int(radius3), (0, 255, 255), 2)  # 2
            cv2.circle(frame_ROI, center3, 5, (255, 255, 255), -1)  # -1
        b3 = np.array([radius3, x3, y3])
        b3_r = np.array([(real_coordinate_x(radius3, 0, tr[0], bl[0], table_real_length - 2*table_real_bumper) + real_coordinate_y(radius3, 0,
                                                                                                         tr[1], bl[1],
                                                                                                         table_real_height - 2*table_real_bumper)) / 2,
                         real_coordinate_x(x3, (tr[0] - bl[0]) / 2, tr[0], bl[0], table_real_length - 2*table_real_bumper),
                         real_coordinate_y(y3, (tr[1] - bl[1]) / 2, tr[1], bl[1], table_real_height - 2*table_real_bumper)])
    x_table = (bl[0]+tr[0])/2
    y_table = (bl[1]+tr[1])/2
    tl = [tr[0], bl[1]]
    br = [bl[0], tr[1]]
    radius_table = math.sqrt((bl[0]-x_table)**2+(bl[1]-y_table)**2)
    center4 = (int((x_table)), int(y_table))
    cv2.circle(frame, center4, 5, (0, 0, 255), -1)
    b4 = np.array([radius_table, x_table, y_table])
    b4_r = np.array(
        [(real_coordinate_x(radius_table, 0, tr[0], bl[0], table_real_length - 2*table_real_bumper) + real_coordinate_y(radius_table, 0, tr[1],
                                                                                              bl[1],
                                                                                              table_real_height - 2*table_real_bumper)) / 2,
         real_coordinate_x(x_table, (tr[0] - bl[0]) / 2, tr[0], bl[0], table_real_length - 2*table_real_bumper),
         real_coordinate_y(y_table, (tr[1] - bl[1]) / 2, tr[1], bl[1], table_real_height - 2*table_real_bumper)])

    frame = cv2.line(frame, (tl[0], tl[1]), (tr[0], tr[1]), (0, 255, 0), 2)
    frame = cv2.line(frame, (tr[0], tr[1]), (br[0], br[1]), (0, 255, 0), 2)
    frame = cv2.line(frame, (br[0], br[1]), (bl[0], bl[1]), (0, 255, 0), 2)
    frame = cv2.line(frame, (bl[0], bl[1]), (tl[0], tl[1]), (0, 255, 0), 2)

    bb = np.hstack([b, b2, b3, b4])
    bb_r = np.hstack([b_r, b2_r, b3_r, b4_r])
    a = np.vstack([a, bb])
    a_r = np.vstack([a_r, bb_r])


def hand_filter():
    global a, current_t, hand_filter_time, bb, time_count, initialization_time, noise_time
    global d_a_r, noise_loss_length

    if current_t <= noise_time:
        d_a_r = a_r[-1] - a_r[-2]
    elif current_t > noise_time:
        d_a_r = a_r[-1] - a_r[-1 - noise_loss_length]

    if round(hand_filter_time, 1) - round(current_t, 1) == 0:
        time_count = time_count + 1
        return
    else:
        hand_filter_time = current_t

        time_count = 0


def billiard_rule():
    global game_num, game_phase, result_array
    global shot_loss_length, shot_param, shot_rate, switch_shot
    global end_loss_length, end_param, end_rate, switch_end
    global current_t, time_shot, time_end, time_shot_length, time_end_length, t0
    global a, a_r, b, b2, b3, b4, bb
    global table_real_length, table_real_height, table_real_bumper
    global raw, stack, folder_directory, file_1_name

    global d_a_r, circle_shape, frame_ROI, bbb, shape_counter_memory, data_length

    if game_num == 0:
        result_array = np.hstack(
            [int(game_num), round(a_r[-1, 1]) / 10, round(-a_r[-1, 2]) / 10, round(a_r[-1, 4]) / 10,
             round(-a_r[-1, 5]) / 10, round(a_r[-1, 7]) / 10, round(-a_r[-1, 8]) / 10, ','])
        dataframe = pd.DataFrame(result_array).T
        print(dataframe)
        dataframe.to_csv(folder_directory + file_1_name, header=False, index=False)
        game_num = 1

    # shot checker part
    if game_phase == 1:

        if len(a) <= shot_loss_length:
            shot_param = 1
        elif len(a) > shot_loss_length:
            shot_param = error_check(a_r[-1 - shot_loss_length, 0], a_r[-1, 0], shot_rate) * error_check(
                a_r[-1 - shot_loss_length, 1], a_r[-1, 1], shot_rate) * error_check(a_r[-1 - shot_loss_length, 2],
                                                                                    a_r[-1, 2],
                                                                                    shot_rate) * error_check(
                a_r[-1 - shot_loss_length, 3], a_r[-1, 3], shot_rate) * error_check(a_r[-1 - shot_loss_length, 4],
                                                                                    a_r[-1, 4],
                                                                                    shot_rate) * error_check(
                a_r[-1 - shot_loss_length, 5], a_r[-1, 5], shot_rate) * error_check(a_r[-1 - shot_loss_length, 6],
                                                                                    a_r[-1, 6],
                                                                                    shot_rate) * error_check(
                a_r[-1 - shot_loss_length, 7], a_r[-1, 7], shot_rate) * error_check(a_r[-1 - shot_loss_length, 8],
                                                                                    a_r[-1, 8],
                                                                                    shot_rate) * error_check(
                a_r[-1 - shot_loss_length, 9], a_r[-1, 9], shot_rate) * error_check(a_r[-1 - shot_loss_length, 10],
                                                                                    a_r[-1, 10],
                                                                                    shot_rate) * error_check(
                a_r[-1 - shot_loss_length, 11], a_r[-1, 11], shot_rate)

        if switch_shot == 0:
            switch_shot = 1
            time_shot = time.time() - t0
        elif switch_shot == 1:
            if shot_param == 0:
                if current_t - time_shot >= time_shot_length:
                    print('Shot detected!')
                    game_phase = 0
                    switch_shot = 0
            elif shot_param == 1:
                time_shot = time.time() - t0
    # end checker part
    elif game_phase == 0:

        if len(a) <= end_loss_length:
            end_param = 0
        elif len(a) > end_loss_length:
            end_param = error_check(a_r[-1 - end_loss_length, 0], a_r[-1, 0], end_rate) * error_check(
                a_r[-1 - end_loss_length, 1], a_r[-1, 1], end_rate) * error_check(a_r[-1 - end_loss_length, 2],
                                                                                  a_r[-1, 2],
                                                                                  end_rate) * error_check(
                a_r[-1 - end_loss_length, 3], a_r[-1, 3], end_rate) * error_check(a_r[-1 - end_loss_length, 4],
                                                                                  a_r[-1, 4],
                                                                                  end_rate) * error_check(
                a_r[-1 - end_loss_length, 5], a_r[-1, 5], end_rate) * error_check(a_r[-1 - end_loss_length, 6],
                                                                                  a_r[-1, 6],
                                                                                  end_rate) * error_check(
                a_r[-1 - end_loss_length, 7], a_r[-1, 7], end_rate) * error_check(a_r[-1 - end_loss_length, 8],
                                                                                  a_r[-1, 8],
                                                                                  end_rate) * error_check(
                a_r[-1 - end_loss_length, 9], a_r[-1, 9], end_rate) * error_check(a_r[-1 - end_loss_length, 10],
                                                                                  a_r[-1, 10],
                                                                                  end_rate) * error_check(
                a_r[-1 - end_loss_length, 11], a_r[-1, 11], end_rate)

        if switch_end == 0:
            switch_end = 1
            time_end = time.time() - t0
        elif switch_end == 1:
            if end_param == 1:
                if current_t - time_end >= time_end_length:
                    print('End detected!')
                    raw = np.hstack([game_num, a[-1, 1], a[-1, 2], a[-1, 4], a[-1, 5], a[-1, 7], a[-1, 8]])
                    white_x = a_r[
                        -1, 1]  # real_coordinate(a[-1, 1], (tr[0] - bl[0]) / 2, tr[0], bl[0], table_real_length)
                    white_y = -a_r[
                        -1, 2]  # -real_coordinate(a[-1, 2], (tr[1] - bl[1]) / 2, tr[1], bl[1], table_real_height)

                    yellow_x = a_r[
                        -1, 4]  # real_coordinate(a[-1, 4], (tr[0] - bl[0]) / 2, tr[0], bl[0], table_real_length)
                    yellow_y = -a_r[
                        -1, 5]  # -real_coordinate(a[-1, 5], (tr[1] - bl[1]) / 2, tr[1], bl[1], table_real_height)

                    red_x = a_r[
                        -1, 7]  # real_coordinate(a[-1, 7], (tr[0] - bl[0]) / 2, tr[0], bl[0], table_real_length)
                    red_y = -a_r[
                        -1, 8]  # -real_coordinate(a[-1, 8], (tr[1] - bl[1]) / 2, tr[1], bl[1], table_real_height)

                    stack = np.hstack(
                        [int(game_num), round(white_x) / 10, round(white_y) / 10, round(yellow_x) / 10,
                         round(yellow_y) / 10,
                         round(red_x) / 10, round(red_y) / 10, ','])
                    print('raw : ', raw)
                    print('stack : ', stack)

                    result_array = np.vstack([result_array, stack])

                    dataframe = pd.DataFrame(result_array)
                    dataframe.to_csv(folder_directory + file_1_name, header=False, index=False)
                    print("white radius : ", radius_ini, " yellow radius : ", radius2_ini, " red radius : ",
                          radius3_ini)

                    if len(a) >= 10 * data_length:
                        print("*********Too Much Information ! :", len(a), "*********")
                        a = a[-data_length:]
                        a_r = a_r[-data_length:]

                    game_num = game_num + 1
                    game_phase = 1
                    switch_end = 0
                else:
                    gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
                    circle_shape = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=90, param2=10,
                                                    minRadius=18, maxRadius=21)
                    print("end_param : ", end_param)
                    shape_counter = 0
                    for i in circle_shape[0]:
                        bbb_x = real_coordinate_x(i[0], (tr[0] - bl[0]) / 2, tr[0], bl[0], table_real_length - 2*table_real_bumper)
                        bbb_y = -real_coordinate_y(i[1], (tr[1] - bl[1]) / 2, tr[1], bl[1], table_real_height - 2*table_real_bumper)
                        bbb_r = real_coordinate_x(i[2], 0, tr[0], bl[0], table_real_length - 2*table_real_bumper)
                        if error_check(a_r[-1, 0], bbb_r, 5) == 1 and error_check(a_r[-1, 1], bbb_x,
                                                                                  30) == 1 and error_check(a_r[-1, 2],
                                                                                                           bbb_y,
                                                                                                           30) == 1:
                            shape_counter_memory_white = shape_counter
                            a_r[-1, 0], a_r[-1, 1], a_r[-1, 2] = circle_shape[shape_counter_memory_white, 2], \
                                                                 circle_shape[shape_counter_memory_white, 0], \
                                                                 circle_shape[shape_counter_memory_white, 1]
                        # if error_check(a_r[-1, 3], bbb_r, 5) == 1 and error_check(a_r[-1, 4], bbb_x, 30) == 1 and error_check(a_r[-1, 5], bbb_y, 30) == 1:
                        #    shape_counter_memory_yellow = shape_counter
                        #    a_r[-1, 3], a_r[-1, 4], a_r[-1, 5] = circle_shape[shape_counter_memory_yellow, 2], circle_shape[shape_counter_memory_yellow, 0], circle_shape[shape_counter_memory_yellow, 1]
                        shape_counter = shape_counter + 1
            elif end_param == 0:
                time_end = time.time() - t0


def show_video():
    global frame_ROI, raw, current_t, file_1_directory, t0, frame
    global kkk, radius_ini, radius2_ini, radius3_ini
    global key_value
    # cv2.circle(frame_ROI, (int(raw[1]), int(raw[2])), int(radius_ini / 2 + radius_ini / 2 * math.sin(current_t * 5)),
    #            (170, 200, 200), -1)
    # cv2.circle(frame_ROI, (int(raw[3]), int(raw[4])), int(radius2_ini / 2 + radius2_ini / 2 * math.sin(current_t * 5)),
    #            (35, 180, 255), -1)
    # cv2.circle(frame_ROI, (int(raw[5]), int(raw[6])), int(radius3_ini / 2 + radius3_ini / 2 * math.sin(current_t * 5)),
    #            (255, 5, 255), -1)
    # cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q") or os.path.isfile(file_1_directory) == False:
        # out_video.write(frame)
        print("The game is finished.")
        key_value = 1
        # close all windows
        return


# output : table tl, tr, bl, br, frame_ROI
def sub_func():
    global file_1_directory, key_value, t0, file_1
    global a, a_r

    variables_setting()
    initialize_table_read()
    initialize_setting()

    while key_value == 0:
        recog_balls()
        billiard_rule()
        show_video()

    vs.release()
    cv2.destroyAllWindows()




