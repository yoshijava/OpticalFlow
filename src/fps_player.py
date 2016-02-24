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

vsync_signal = 1/float(60)
scale = 0.4
DEBUG = True
SIMULATE_VSYNC = True

# Everything we do here is treated as overheadless
def blackhole_operation(img):
    t1 = time.time()
    img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t2 = time.time()
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        return None, None
    return (t2-t1), img

def flush(msg):
    if DEBUG:
        print(msg)
        sys.stdout.flush()

if __name__ == '__main__':
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
    try:
        # process the video with "raw speed" assuming it's far faster than 1/target_fps
        # if the raw speed is not fast enough, it causes "slow motion" when displaying (which is not equivalent to frame dropping)
        while True:
            t1 = time.time()
            nFrames += 1
            try:
                img = vid.get_data(i)
                i+=1
            except IndexError:
                break
            overhead, img = blackhole_operation(img)
            if overhead == None:
                break
            now = time.time();
            period = (now - start_time)
            if period >= (1/target_fps)-overhead:
                cv2.imshow('FPS player', img)
                flush("Instant FPS = " + str(60*(nFrames - skippedFrames)/float(nFrames)))
                start_time = now
            else:
                skippedFrames += 1

            t2 = time.time()
            elapsed = t2 - t1

            if SIMULATE_VSYNC:
                if elapsed <= vsync_signal :
                    # for avoiding "fast motion"
                    slack = vsync_signal - elapsed
                    flush("sleep for " + str(slack) + " seconds.")
                    time.sleep(slack)

    except KeyboardInterrupt:
        print ("Interrupted by user...")

    cv2.destroyAllWindows()
    print("nFrames = " , nFrames)
    print("skippedFrames =", skippedFrames)
    print("Avg FPS = ", 60*(nFrames - skippedFrames)/float(nFrames))
