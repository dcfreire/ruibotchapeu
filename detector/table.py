import cv2
import numpy as np


class Table:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.size = (600, 1000)
        frame = self.cap.read()[1]
        self.homography = self.get_homography(frame, self.size)
        self.lower, self.upper = self._get_cloth_range(frame)

    def start(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.warpPerspective(frame, self.homography, self.size)
            mask = self._get_cloth_mask(frame)
            ctrs = self._get_ball_contours(mask)
            for c in ctrs:
                x, y, w, z = cv2.boundingRect(c)
                x -= 6
                y -= 6
                w += 10
                z += 10
                cv2.rectangle(frame, (x, y), (x + w, y + z), (0, 0, 255), 2)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _get_contour(self, img):
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0]
        c = max(cnts, key=cv2.contourArea)
        corners = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        return corners, c

    def _order_points(self, points):
        center = np.mean(points)
        shifted = points - center
        theta = np.arctan2(shifted[:, 0], shifted[:, 1])
        ind = np.argsort(theta)
        return points[ind]

    def get_homography(self, frame, size=(500, 1000, 3)):

        # get the corners of the table
        frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        corners, _ = self._get_contour(th)
        pts_dst = np.array(
            [[0, 0], [size[0] - 1, 0], [size[0] - 1, size[1] - 1], [0, size[1] - 1]],
            dtype=float,
        )
        corners = self._order_points(corners[:, 0])
        pts_dst = self._order_points(pts_dst)
        h, _ = cv2.findHomography(corners, pts_dst)
        return h



    def _get_unique_hsv(self, frame):
        h, s, v = cv2.split(
            cv2.cvtColor(
                cv2.warpPerspective(frame, self.homography, self.size),
                cv2.COLOR_BGR2HSV,
            )
        )

        # get uniques
        unique_h, counts_h = np.unique(h, return_counts=True)
        unique_s, counts_s = np.unique(s, return_counts=True)
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

        return big_h, big_s, big_v

    def _get_cloth_range(self, frame):
        big_h, big_s, big_v = self._get_unique_hsv(frame)

        # get the color mask
        marginh, margin, marginv = 10, 80, 150
        lowerb = np.array([big_h - marginh, big_s - margin, big_v - marginv])
        upperb = np.array([big_h + marginh, big_s + margin, big_v + marginv])

        return lowerb, upperb

    def _get_cloth_mask(self, frame):
        frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower, self.upper)

        kernel = np.ones((5, 5), np.uint8)
        mask_closing = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel
        )  # Cleans inner points

        # Applies the mask to the original frame
        _, mask_inv = cv2.threshold(mask_closing, 5, 255, cv2.THRESH_BINARY_INV)

        return mask_inv

    def _get_ball_contours(self, mask):
        ctrs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return [
            ctr
            for ctr in ctrs
            if 200 < cv2.contourArea(ctr) < 800
            and cv2.minAreaRect(ctr)[1][0] * 0.15 < cv2.minAreaRect(ctr)[1][1]
            and cv2.minAreaRect(ctr)[1][1] * 0.15 < cv2.minAreaRect(ctr)[1][0]
        ]
