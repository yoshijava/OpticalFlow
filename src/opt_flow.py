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

interval = 1
max_vector_dist = [0]*35

def get_flow_lines(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    global max_vector_dist
    max_vector_dist = get_max_vector_dist(lines)
    return lines

def get_max_vector_dist(lines):
    max_dist = 0
    for (x1, y1), (x2, y2) in lines:
        dist = (x1-x2)^2 + (y1-y2)^2
        if max_dist < dist:
            max_dist = dist
    # if max_dist != 0:
    global scale
    sqrt_max_dist = math.sqrt(max_dist)/scale
    # print("max dist = %d" % (sqrt_max_dist/scale)),
    max_index = int(sqrt_max_dist)
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
    if(len(sys.argv)<2):
        print ("python opt_flow.py <<filename.mp4>> <<scale>>")
        exit()
    filename = sys.argv[1]
    scale = float(sys.argv[2])
    vid = imageio.get_reader(filename,  'ffmpeg')
    prev = vid.get_data(0)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev = cv2.resize(prev, (0,0), fx=scale, fy=scale)
    while True:
        start_time = time.time()
        i = i + interval
        try:
            img = vid.get_data(i)
        except IndexError:
            break
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            flow = cv2.calcOpticalFlowFarneback(prev, img, None, 0.5, # pyr_scale
                3, # levels
                16, # window size
                1, # iterations
                5, # poly_n
                1.1, # poly_sigma
                0 # flag
                )
            prev = img
            cv2.imshow('flow', draw_flow(img, flow))

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            period = (time.time() - start_time)
            # print("--- Process time: %s sec ---" % period)
    cv2.destroyAllWindows()
    plt.plot(max_vector_dist)
    plt.show()
