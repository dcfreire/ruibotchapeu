import numpy as np
import cv2

class Tracker:
    def __init__(self):
        self.contours = []
    
    def track(self, frame1, frame2):
        """
            Has a 1 frame delay and can track only moving balls
            Contours will not necessarilly start with white ball contour
        """
        self.contours = []
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, th = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5))
        dilated = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=5)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 50:
                self.contours.append(contour)