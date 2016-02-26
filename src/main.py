#!/usr/bin/env python

'''
example to show optical flow

USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import cv2
import video
import time
import sys
import math
import imageio
from subprocess import call

DEBUG = True
interval = 1
max_vector_dist = [0]
avg_vector_dist = [0]

def flush(msg):
    if DEBUG:
        print(msg)
        sys.stdout.flush()

def get_flow_lines(img, flow, step=16):
    global max_vector_dist
    global avg_vector_dist
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    max_vector_dist = get_max_vector_dist(lines)
    avg_vector_dist = get_avg_vector_dist(lines)
    return lines

def get_avg_vector_dist(lines):
    global avg_vector_dist
    sum_vector = 0
    for (x1, y1), (x2, y2) in lines:
        sum_vector += math.sqrt((x1-x2)**2 + (y1-y2)**2)
    avg_index = int(sum_vector/len(lines))
    while avg_index > len(avg_vector_dist)-1:
        less = avg_index - len(avg_vector_dist) + 1
        avg_vector_dist += ([0] * less)
    avg_vector_dist[avg_index] += 1
    print (avg_vector_dist)
    sys.stdout.flush()
    return avg_vector_dist

def get_max_vector_dist(lines):
    global max_vector_dist
    max_dist = 0
    for (x1, y1), (x2, y2) in lines:
        dist = (x1-x2)**2 + (y1-y2)**2
        if max_dist < dist:
            max_dist = dist
    # if max_dist != 0:
    sqrt_max_dist = math.sqrt(max_dist)
    max_index = int(sqrt_max_dist)
    # dynamic extend the size of list
    while max_index > len(max_vector_dist)-1:
        less = max_index - len(max_vector_dist) + 1
        max_vector_dist += ([0] * less)
    # print(len(max_vector_dist))
    max_vector_dist[max_index] += 1
    print (max_vector_dist)
    sys.stdout.flush()
    return max_vector_dist

def draw_flow(img, flow, step=16):
    lines = get_flow_lines(img, flow, step)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

if __name__ == '__main__':
    i = 1
    if(len(sys.argv)<3):
        print ("python opt_flow.py <<filename.mp4>> <<scale>> <<target_fps>>")
        exit()
    filename = sys.argv[1]
    scale = float(sys.argv[2])
    target_fps = float(sys.argv[3])

    vid = imageio.get_reader(filename,  'ffmpeg')
    fps = vid.get_meta_data()['fps']
    vsync_signal = 1/fps
    if target_fps == None:
        target_fps = fps
    flush("The FPS of this video is recorded as " + str(fps))
    if target_fps > fps:
        print("The FPS you requested is higher than recorded FPS, which will result in slow motion.")
    img = vid.get_data(0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (0,0), fx=scale, fy=scale)

    try:
        wall_clock_t1 = time.time()
        while True:
            t1 = time.time()
            try:
                prev = img
                img = vid.get_data(int(i))
                i += fps/target_fps
            except IndexError:
                break

            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            flush("Frame to fetch = " + str(int(i)))
            flow = cv2.calcOpticalFlowFarneback(prev, img, None, 0.5, # pyr_scale
                3, # levels
                16, # window size
                1, # iterations
                5, # poly_n
                1.1, # poly_sigma
                0 # flag
                )
            prev = img
            draw_flow(img, flow)
            # cv2.imshow('flow', draw_flow(img, flow))

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break


        wall_clock_t2 = time.time()
    except KeyboardInterrupt:
        print ("Interrupted by user...")

    print("wall clock elapsed = ", (wall_clock_t2-wall_clock_t1), " sec")
    cv2.destroyAllWindows()

    avg_vector_dist.extend([0] * (len(max_vector_dist) - len(avg_vector_dist)))
    plt.figure(1)
    plt.subplot(211)
    plt.xlabel('vector length')
    plt.ylabel('# of vectors')
    plt.title('Max vector distribution')
    plt.plot(max_vector_dist)
    plt.subplot(212)
    plt.title('Avg vector distribution')
    plt.ylabel('# of vectors')
    plt.xlabel('vector length')
    plt.plot(avg_vector_dist)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()