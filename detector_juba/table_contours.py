import cv2
import numpy as np

def get_contours(img):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0]
    c = max(cnts, key=cv2.contourArea)
    corners = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    return corners, c

def order_points(points):
    center = np.mean(points)
    shifted = points - center
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])
    ind = np.argsort(theta)
    return points[ind]

def get_homography(frame, size=(500, 1000, 3)):
    frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    corners, _ = get_contours(th)
    pts_dst = np.array(
               [
                [0,0],
                [size[0] - 1, 0],
                [size[0] - 1, size[1] -1],
                [0, size[1] - 1 ]
                ], dtype=float
               )
    corners = order_points(corners[:, 0])
    pts_dst = order_points(pts_dst)
    h, _ = cv2.findHomography(corners, pts_dst)
    return h

def get_cloth_range(h, s, v):
    # get uniques
    unique_s, counts_s = np.unique(s, return_counts=True)
    unique_h, counts_h = np.unique(h, return_counts=True)
    unique_v, counts_v = np.unique(v, return_counts=True)

    # sort through and grab the most abundant unique color
    big_s, big_h, big_v = 0, 0, 0
    biggest_s, biggest_h, biggest_v = -1, -1, -1

    for i, curs in enumerate(unique_s):
        if counts_s[i] > biggest_s:
            biggest_s = counts_s[i]
            big_s = int(curs)

    for i, curh in enumerate(unique_h):
        if counts_h[i] > biggest_h:
            biggest_h = counts_h[i]
            big_h = int(curh)

    for i, curv in enumerate(unique_v):
        if counts_v[i] > biggest_v:
            biggest_v = counts_v[i]
            big_v = int(curv)

    # get the color mask
    marginh, margin, marginv = 10, 80, 150
    lowerb = np.array([big_h - marginh, big_s - margin, big_v - marginv])
    upperb = np.array([big_h + marginh, big_s + margin, big_v + marginv])
    return lowerb, upperb

def get_cloth_mask(frame, lowerb=None, upperb=None):
    frame = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if lowerb is None or upperb is None:
        lowerb, upperb = get_cloth_range(h, s, v)

    mask = cv2.inRange(hsv, lowerb, upperb)

    kernel = np.ones((5,5), np.uint8)
    mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Cleans inner points

    # Applies the mask to the original frame
    _,mask_inv = cv2.threshold(mask_closing,5,255,cv2.THRESH_BINARY_INV)
    #masked_img = cv2.bitwise_and(frame,frame, mask=mask_inv)
    return mask_inv

def get_ball_contours(mask):
    ctrs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [ctr for ctr in ctrs
                        if 200 < cv2.contourArea(ctr) < 800
                        and cv2.minAreaRect(ctr)[1][0]*0.15 < cv2.minAreaRect(ctr)[1][1]
                        and cv2.minAreaRect(ctr)[1][1]*0.15 < cv2.minAreaRect(ctr)[1][0]
            ]
