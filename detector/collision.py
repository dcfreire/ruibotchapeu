import cv2
import numpy as np

def check_collision(contours): 
    """ Checks if a collision has happened based on euclidean distance between contour points.
    If euclidean distance is less than 50 (arbitrary threshold), then there was a collision

    Parameters:
    -----------
        contours: List
            list of contours, in which white ball contour should be the first element

    Returns:
    --------
        collision: Boolean
            collision occurred / not ocurred - True / False

    """
    collision = False

    if len(contours)>1:
        white = np.array(contours[0])
        colors = np.array(contours[1::])

        # Squared difference between contours coordinates
        srdq = (white[:, None, :] - colors)**2
        # Euclidean distance between every white ball and colored balls contour points
        distances = np.sqrt(np.sum(srdq, axis=3))
        if (distances < 50).any():
            #!TODO check if this threshold really works for official tracking
            collision = True

    return collision