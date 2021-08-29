import numpy as np
import cv2

class Tracker:
    def __init__(self):
        self.contours = []
        self.movement = []
    
    def track(self, frame1, frame2, masked = False):
        """
            Has a 1 frame delay and can track only moving balls
            Contours will not necessarilly start with white ball contour
        """
        self.contours = []
        frame1 = frame1.copy()
        frame2 = frame2.copy()
        if masked:
            frame_ycrcb = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)
            frame2_ycrcb = cv2.cvtColor(frame2, cv2.COLOR_BGR2YCrCb)
            
            # equalize the histogram of the Y channel
            frame_ycrcb[:,:,0] = cv2.equalizeHist(frame_ycrcb[:,:,0])
            frame2_ycrcb[:,:,0] = cv2.equalizeHist(frame2_ycrcb[:,:,0])
            
            # convert the YUV image back to RGB format
            frame1 = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCrCb2BGR)
            frame2 = cv2.cvtColor(frame2_ycrcb, cv2.COLOR_YCrCb2BGR)
            frame1 = cv2.cvtColor(frame_ycrcb, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2_ycrcb, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(frame1, frame2)
            diff = cv2.normalize(diff, diff, 0, 255, cv2.NORM_MINMAX)
            gray = diff.copy()
        else:
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        self.movement = gray
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, th = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5))
        dilated = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=5)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 80:
                self.contours.append(contour)