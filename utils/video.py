import cv2

def saveVideo(frames, size=(640, 480)):
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    out = cv2.VideoWriter('output.mp4', fourcc, 30, size)
    for frame in frames:
        out.write(frame)
    out.release()