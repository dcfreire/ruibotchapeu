import cv2
import numpy as np
from tracker.movement_tracker import Tracker

def check_collision(contours): 
    """ Checks if a collision has happened based on euclidean distance between contour points.
    If euclidean distance is less than 60 (arbitrary threshold), then there was a collision

    Parameters:
    -----------
        contours: List
            list of contours, in which white ball contour should be the first element

    Returns:
    --------
        collision: Boolean
            collision occurred / not ocurred - True / False
        col_dir: np.array
            normalized collision direction
        col_pos: ndarray
            collision's pixel coordinates

    """
    collision = False
    col_dir = []
    col_pos = []

    if len(contours)>1:
        white = np.array(contours[0])
        for idx, ctr in enumerate(contours[1::]):
            colors = ctr.copy()
        
            # Squared difference between contours coordinates
            srdq = (white[:, None, :] - colors)**2
            
            # Euclidean distance between every white ball and colored balls contour points
            distances = np.sqrt(np.sum(srdq, axis=3))
            dist_min = 60

            if (distances < dist_min).any():
                col_pos.append(white[np.nonzero((distances < dist_min))[0][0]])

                M = cv2.moments(white)
                cx_white = int(M['m10']/M['m00'])
                cy_white = int(M['m01']/M['m00'])
                Mc = cv2.moments(colors)
                cx_color = int(Mc['m10']/Mc['m00'])
                cy_color = int(Mc['m01']/Mc['m00'])

                direction = np.array([cx_white - cx_color, cy_white - cy_color])
                direction = direction / np.linalg.norm(direction)
                col_dir.append(direction)
                collision = True
                
    return (collision, col_dir, col_pos)

def collision_checker(table):
    homography = table.homography
    size = table.size

    ret, frame = table.cap.read()
    frame = cv2.warpPerspective(frame, homography, size[:2])

    ret2, frame2 = table.cap.read()
    frame2 = cv2.warpPerspective(frame2, homography, size[:2])
   
    first_frame = frame.copy()
    first_frame2 = frame.copy()
    coll_num = 0
    mask_collision = np.zeros_like(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY))
    while table.cap.isOpened():
        if not ret2:
            break

        tracker = Tracker()
        tracker.track(frame, frame2)
        contours = tracker.contours
        
        cv2.drawContours(frame, contours, -1, (255,0,0), 2)
        collision, col_dir, col_pos = check_collision(contours)
        color_name = ''

        if collision:
            coll_num += 1
            cont = contours[0]
            M = cv2.moments(cont)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            radius = 20
            center = tuple((col_pos[0].ravel() - col_dir[0]*2*radius).astype(np.int32))
            cv2.ellipse(mask_collision, center, (int(radius*1.25), radius),90,0,360,255,-1)
            cv2.putText(frame, 'TEI!',(cx-100,cy-100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5, cv2.LINE_AA)
            
            mask = table._get_cloth_mask(first_frame)
            mask = cv2.bitwise_and(mask, mask, mask=mask_collision)
            first_frame2 = mask_collision.copy()
            ctrs = table._get_ball_contours(mask)
            
            for c in ctrs:
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = table.get_common_color(first_frame, c)
                color_name = table.predict_color(color)
                print(color_name)
                x, y, w, z = cv2.boundingRect(c)
                x -= 6
                y -= 6
                w += 10
                z += 10
                cv2.putText(first_frame, f'{color_name}' , (x, y-5), font, 0.5, (int(color[0]), int(color[1]), int(color[2])), 2, cv2.LINE_AA)
                cv2.rectangle(first_frame, (x, y), (x + w, y + z), (int(color[0]), int(color[1]), int(color[2])), 2)
        cv2.putText(frame, '#Collisions: {}'.format(coll_num),(130,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
        
        frame = frame2
        ret2, frame2 = table.cap.read()
        if ret2:
            frame2 = cv2.warpPerspective(frame2, homography, size[:2])
        yield (collision, col_pos, col_dir, color_name, first_frame2)