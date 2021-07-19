import cv2
import numpy as np

def get_contours(img):
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    corners = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    return corners, c

def order_points(points):
    center = np.mean(points)
    shifted = points - center
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])
    ind = np.argsort(theta)
    return points[ind]