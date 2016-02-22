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

def flush():
    sys.stdout.flush()

if __name__ == '__main__':
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    targetFps = float(sys.argv[2])
    filename = "../videos/" + sys.argv[1]
    vid = imageio.get_reader(filename,  'ffmpeg')
    start_time = time.time()
    i = 0
    while True:
        overhead_t1 = time.time()
        img = vid.get_data(i)
        # print (t2-t1)
        i+=1
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
        overhead_t2 = time.time()
        now = time.time();
        period = (now - start_time)
        if period >= (1/targetFps)-(overhead_t2-overhead_t1):
            cv2.imshow('FPS player', img)
            print ("FPS = ", (1/period))
            flush()
            start_time = now

    cv2.destroyAllWindows()
