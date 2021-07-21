import cv2
import numpy as np

def get_contours(img):
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    corners = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    return corners, c

def order_points(points):
    center = np.mean(points)
    shifted = points - center
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])
    ind = np.argsort(theta)
    return points[ind]

def get_homography(frame, size=(500, 1000, 3)):
    frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5,5), 1)
    frame = cv2.Canny(frame, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    pts_src, _ = get_contours(frame)
    pts_dst = np.array(
                   [
                    [0,0],
                    [size[0] - 1, 0],
                    [size[0] - 1, size[1] -1],
                    [0, size[1] - 1 ]
                    ], dtype=float
                   )
    pts_src = order_points(pts_src[:, 0])
    pts_dst = order_points(pts_dst)
    h, _ = cv2.findHomography(pts_src, pts_dst)
    return h
