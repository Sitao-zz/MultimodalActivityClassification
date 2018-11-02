import numpy as np
import scipy.io as sio
import cv2
import pickle as pk
from common.dataprep import definitions

dataset, trainsets, validationsets = definitions()
depth_dataset = {i: {'depth': sio.loadmat('dataset/Depth/' + i + '_depth.mat')['d_depth']} for i in dataset}


def processRGBData(cap, sub):
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    framenum = 1

    ret, frame1 = cap.read()

    depth = depth_dataset[sub]['depth'][:, :, 5]
    depth = depth.astype(np.uint8)
    depth = cv2.threshold(depth, 0, 255, cv2.THRESH_BINARY)[1]
    depth = cv2.resize(depth, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    frame1 = cv2.bitwise_and(frame1, frame1, mask=depth)

    x, y, w, h = 170, 30, 300, 450
    rect = cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 0), 1)
    frame1 = rect[y:y + h, x:x + w]
    frame1 = cv2.medianBlur(frame1, 7)

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    hof = []

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:

            depth = depth_dataset[sub]['depth'][:, :, np.clip(framenum + 5, 0, 40)]
            depth = depth.astype(np.uint8)
            depth = cv2.threshold(depth, 0, 255, cv2.THRESH_BINARY)[1]
            depth = cv2.resize(depth, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            frame = cv2.bitwise_and(frame, frame, mask=depth)
            x, y, w, h = 170, 30, 300, 450
            rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)
            frame = rect[y:y + h, x:x + w]

            frame = cv2.medianBlur(frame, 7)

            nex = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs, nex, None, pyr_scale=0.5,
                                                levels=5, winsize=15, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)

            flow_rsz = cv2.resize(flow, (48, 48))

            mag, ang = cv2.cartToPolar(flow_rsz[..., 0], flow_rsz[..., 1])
            _mag, _ang = [], []
            for r in mag:
                for c in r:
                    _mag.append(c)
            for r in ang:
                for c in r:
                    _ang.append(c)
            hof.append(_mag + _ang)

            if framenum % 3 == 1:
                prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        else:
            break
        framenum = framenum + 1

    cap.release()
    return hof


hofset = []
for i in dataset:
    print("processing " + i)
    cap = cv2.VideoCapture('dataset/RGB/' + i + '_color.avi')
    hofset.append(processRGBData(cap, i))

pk.dump(hofset, open("hofset.pk", "wb"))
