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

scale = 0.4
DEBUG = True

def flush(msg):
    if DEBUG:
        print(msg)
        sys.stdout.flush()

if __name__ == '__main__':
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0
    if(len(sys.argv)<3):
        target_fps = 60
    else:
        target_fps = float(sys.argv[2])
    flush(sys.argv)
    filename = sys.argv[1]
    vid = imageio.get_reader(filename,  'ffmpeg')
    start_time = time.time()
    i = 0
    nFrames = 0
    skippedFrames = 0
    while True:
        overhead_t1 = time.time()
        nFrames += 1
        try:
            img = vid.get_data(i)
        except IndexError:
            break
        else:
            # print (t2-t1)
            i+=1
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            overhead_t2 = time.time()
            now = time.time();
            period = (now - start_time)
            if period >= (1/target_fps)-(overhead_t2-overhead_t1):
                cv2.imshow('FPS player', img)
                flush("Instant FPS = " + str(60*(nFrames - skippedFrames)/float(nFrames)))
                start_time = now
            else:
                skippedFrames += 1
    cv2.destroyAllWindows()
    print("nFrames = " , nFrames)
    print("skippedFrames =", skippedFrames)
    print("Avg FPS = ", 60*(nFrames - skippedFrames)/float(nFrames))
