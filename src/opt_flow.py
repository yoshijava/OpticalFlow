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

import numpy as np
import cv2
import video
import time
import sys
import math
import pylab
import imageio
from subprocess import call

scale = 0.3
interval = 1
vectorDist = [0]*35

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    maxDist = 0
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        dist = (x1-x2)^2 + (y1-y2)^2
        if maxDist < dist:
            maxDist = dist
    # if maxDist != 0:
    global scale
    sqrtMaxDist = math.sqrt(maxDist)/scale
    # print("max dist = %d" % (sqrtMaxDist/scale)),
    index = int(sqrtMaxDist)
    global vectorDist
    vectorDist[index] += 1
    print (vectorDist)
    sys.stdout.flush()
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    i = 1
    filename = sys.argv[1]
    vid = imageio.get_reader(filename,  'ffmpeg')
    prev = vid.get_data(0)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev = cv2.resize(prev, (0,0), fx=scale, fy=scale)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev
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
            if show_hsv:
                cv2.imshow('flow HSV', draw_hsv(flow))
            if show_glitch:
                cur_glitch = warp_flow(cur_glitch, flow)
                cv2.imshow('glitch', cur_glitch)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            if ch == ord('1'):
                show_hsv = not show_hsv
                print('HSV flow visualization is', ['off', 'on'][show_hsv])
            if ch == ord('2'):
                show_glitch = not show_glitch
                if show_glitch:
                    cur_glitch = img
                print('glitch is', ['off', 'on'][show_glitch])
            period = (time.time() - start_time)
            # print("--- Process time: %s sec ---" % period)
    cv2.destroyAllWindows()
