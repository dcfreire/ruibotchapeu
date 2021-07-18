class MultiTracker:
    def __init__(self):
        self.trackers = []

    def add(self, tracker, frame, roi):
        tracker.init(frame, roi)
        self.trackers.append(tracker)
    
    def update(self, frame):
        rois = []
        for tracker in self.trackers:
            _, roi = tracker.update(frame)
            rois.append(roi)
        return rois

    