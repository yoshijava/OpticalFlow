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
scale = 0.1
DEBUG = False
SIMULATE_VSYNC = True

# Everything we do here is treated as overheadless
def blackhole_operation(img):
    img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

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
    framePlayed = 0
    vsync_overhead = 0
    try:
        # process the video with "raw speed" assuming it's far faster than 1/target_fps
        # if the raw speed is not fast enough, it causes "slow motion" when displaying (which is not equivalent to frame dropping)
        while True:
            t1 = time.time()
            try:
                img = vid.get_data(i)
                i+=1
            except IndexError:
                break

            nFrames += 1
            overhead_start = time.time()
            img = blackhole_operation(img)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            overhead_end = time.time()
            overhead = overhead_end - overhead_start + vsync_overhead

            now = time.time();
            period = (now - start_time)
            if period >= (1/target_fps)-overhead:
                framePlayed += 1
                cv2.imshow('FPS player', img)
                flush("Instant FPS = " + str(60*framePlayed/float(nFrames)))
                start_time = now

            t2 = time.time()
            elapsed = t2 - t1

            if SIMULATE_VSYNC:
                vsync_overhead_start = time.time()
                if elapsed <= vsync_signal :
                    # for avoiding "fast motion"
                    slack = vsync_signal - elapsed
                    flush("sleep for " + str(slack) + " seconds.")
                    time.sleep(slack)
                vsync_overhead_end = time.time()
                vsync_overhead = vsync_overhead_end - vsync_overhead_start
                flush("vsync overhead = " + str(vsync_overhead))

    except KeyboardInterrupt:
        print ("Interrupted by user...")

    cv2.destroyAllWindows()
    print("nFrames = " , nFrames)
    print("skippedFrames =", nFrames-framePlayed)
    print("Avg FPS = ", 60*framePlayed/float(nFrames))
