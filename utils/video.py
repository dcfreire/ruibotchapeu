import cv2

def saveVideo(frames):
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    out = cv2.VideoWriter('output.mp4', fourcc, 30, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()