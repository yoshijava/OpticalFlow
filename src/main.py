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
# from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import cv2
import video
import time
import sys
import math
import imageio
from threading import Timer

DEBUG = False
interval = 1
max_vector_dist = [0]
avg_vector_dist = [0]
frame_vec_max = []
frame_vec_avg = []
plt_update_flag = True
frame = 0
max_x1 = -1
max_y1 = -1

def flush(msg):
    if DEBUG:
        print msg
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

    frame_vec_avg.append(avg_index)

    flush(avg_vector_dist)
    sys.stdout.flush()
    return avg_vector_dist

def get_max_vector_dist(lines):
    global max_vector_dist
    global max_x1
    global max_y1
    max_dist = 0
    for (x1, y1), (x2, y2) in lines:
        dist = (x1-x2)**2 + (y1-y2)**2
        if max_dist < dist:
            max_dist = dist
            max_x1, max_y1 = x1, y1
    # if max_dist != 0:
    sqrt_max_dist = math.sqrt(max_dist)
    max_index = int(sqrt_max_dist)
    # dynamic extend the size of list
    while max_index > len(max_vector_dist)-1:
        less = max_index - len(max_vector_dist) + 1
        max_vector_dist += ([0] * less)
    # print len(max_vector_dist)
    max_vector_dist[max_index] += 1

    frame_vec_max.append(max_index)

    flush(max_vector_dist)
    sys.stdout.flush()
    return max_vector_dist

def draw_flow(img, flow, step=16):
    global max_x1
    global max_y1
    lines = get_flow_lines(img, flow, step)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        if x1 == max_x1 and y1 == max_y1:
            cv2.circle(vis, (x1, y1), 5, (0, 0, 255), -1)
        else:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

# def draw_hsv(flow):
#     h, w = flow.shape[:2]
#     fx, fy = flow[:,:,0], flow[:,:,1]
#     ang = np.arctan2(fy, fx) + np.pi
#     v = np.sqrt(fx*fx+fy*fy)
#     hsv = np.zeros((h, w, 3), np.uint8)
#     hsv[...,0] = ang*(180/np.pi/2)
#     hsv[...,1] = 255
#     hsv[...,2] = np.minimum(v*4, 255)
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     return bgr


# def warp_flow(img, flow):
#     h, w = flow.shape[:2]
#     flow = -flow
#     flow[:,:,0] += np.arange(w)
#     flow[:,:,1] += np.arange(h)[:,np.newaxis]
#     res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
#     return res

def init_figure():
    plt_update_flag = True
    avg_vector_dist.extend([0] * (len(max_vector_dist) - len(avg_vector_dist)))
    plt.figure(1,figsize=(20,10))
    plt.ion()
    # plt.subplot(211)
    # plt.plot(max_vector_dist, lw=5, color='red')
    # plt.subplot(212)
    # plt.plot(avg_vector_dist, lw=5, color='green')
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')

    plt.show()

def plt_update():
    while plt_update_flag:
        # avg_vector_dist.extend([0] * (len(max_vector_dist) - len(avg_vector_dist)))
        # xlim = max(len(max_vector_dist), len(avg_vector_dist))
        plt.subplot(221)
        plt.cla()
        plt.title('Max vector timeline')
        plt.ylabel('Vector size')
        plt.xlabel('timeline')
        plt.plot(frame_vec_max,lw=3,color='red')

        plt.subplot(222)
        plt.cla()
        plt.title('Max vector distribution')
        plt.xlabel('vector length')
        plt.ylabel('# of vectors')
        plt.plot(max_vector_dist, lw=3, color='red')

        plt.subplot(223)
        plt.cla()
        plt.title('Avg vector timeline')
        plt.ylabel('Vector size')
        plt.xlabel('timeline')
        plt.plot(frame_vec_avg,lw=3,color='green')

        plt.subplot(224)
        plt.cla()
        plt.title('Avg vector distribution')
        plt.ylabel('# of vectors')
        plt.xlabel('vector length')
        plt.plot(avg_vector_dist, lw=3, color='green')

        plt.draw()
        plt.pause(0.5)

if __name__ == '__main__':

    print str(sys.argv)
    if(len(sys.argv)<3):
        print "python main.py <<filename.mp4>> <<scale>> (target_fps) (output.mp4)"
        print "Arguments enclosed by () are optional."
        exit()
    filename = sys.argv[1]
    scale = float(sys.argv[2])

    vid = imageio.get_reader(filename,  'ffmpeg')
    fps = vid._meta['fps']
    num_frames = vid._meta['nframes']
    print "# of frames = " + str(num_frames)
    print "The FPS of this video is recorded as " + str(fps)
    vsync_signal = 1/fps
    if(len(sys.argv)>=4):
        target_fps = float(sys.argv[3])
        print "target fps is set to " + str(target_fps)
        if target_fps > fps:
            print "The FPS you requested is higher than recorded FPS, which will result in slow motion."
    else:
        target_fps = fps
        print "Set to recorded FPS."
    img = vid.get_data(0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (0,0), fx=scale, fy=scale)

    write_to_video = False
    if(len(sys.argv)>=5):
        out_filename = sys.argv[4]
        writer = imageio.get_writer(out_filename, fps=target_fps)
        write_to_video = True

    init_figure()
    t = Timer(0, plt_update)
    t.start()

    show_hsv = False
    show_glitch = False

    try:
        wall_clock_t1 = time.time()
        for frame in range(num_frames):
            t1 = time.time()

            prev = img
            flush("Frame to fetch = " + str(int(frame)))
            img = vid.get_data(int(frame))

            timestamp = float(frame)/ vid.get_meta_data()['fps']
            # print("Frame " + str(frame) + "'s timestamp = " + str(timestamp))

            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev, img, None, 0.5, # pyr_scale
                3, # levels
                16, # window size
                1, # iterations
                5, # poly_n
                1.1, # poly_sigma
                0 # flag
                )
            prev = img
            img_mv = draw_flow(img, flow)
            cv2.imshow('flow', img_mv)
            if write_to_video == True:
                writer.append_data(img_mv)

            print str(int(frame)) + " -> " + str(frame_vec_max[int(frame)])
            frame += fps/target_fps

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
            # if ch == ord('1'):
            #     show_hsv = not show_hsv
            #     print('HSV flow visualization is', ['off', 'on'][show_hsv])
            # if ch == ord('2'):
            #     show_glitch = not show_glitch
            #     if show_glitch:
            #         cur_glitch = img
            #     print('glitch is', ['off', 'on'][show_glitch])
            #
            # if show_hsv:
            #     cv2.imshow('flow HSV', draw_hsv(flow))
            # if show_glitch:
            #     cur_glitch = warp_flow(cur_glitch, flow)
            #     cv2.imshow('glitch', cur_glitch)

        wall_clock_t2 = time.time()
    except KeyboardInterrupt:
        print "Interrupted by user..."

    if write_to_video == True:
        writer.close()

    plt_update_flag = False
    cv2.destroyWindow('flow')
    print "wall clock elapsed = ", (wall_clock_t2-wall_clock_t1), " sec"

    # print frame_vec_avg
    # t.cancel()
    plt.show(block=True)
