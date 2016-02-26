#!/usr/bin/env python

'''
FPS player

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

# vsync_signal = 1/float(60)
scale = 0.4
DEBUG = True

def flush(msg):
    if DEBUG:
        print(msg)
        sys.stdout.flush()

if __name__ == '__main__':
    if(len(sys.argv)<3):
        target_fps = None
    else:
        target_fps = float(sys.argv[2])

    flush(sys.argv)
    filename = sys.argv[1]
    vid = imageio.get_reader(filename,  'ffmpeg')
    start_time = time.time()
    i = float(0)
    vsync_overhead = 0
    new_refresh = 0

    fps = vid.get_meta_data()['fps']
    vsync_signal = 1/fps
    if target_fps == None:
        target_fps = fps
    flush("The FPS of this video is recorded as " + str(fps))
    if target_fps > fps:
        print("The FPS you requested is higher than recorded FPS, which will result in slow motion.")

    try:
        wall_clock_t1 = time.time()
        while True:
            t1 = time.time()
            try:
                img = vid.get_data(int(i))
                i += fps/target_fps
            except IndexError:
                break

            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

            flush("Frame to fetch = " + str(int(i)))
            cv2.imshow('FPS player', img)
            last_refresh = new_refresh
            new_refresh = time.time()
            t2 = time.time()
            if (t2-t1) < 1/target_fps:
                time.sleep( 1/target_fps - (t2-t1))

        wall_clock_t2 = time.time()
    except KeyboardInterrupt:
        print ("Interrupted by user...")

    cv2.destroyAllWindows()
    print("wall clock elapsed = ", (wall_clock_t2-wall_clock_t1), " sec")
