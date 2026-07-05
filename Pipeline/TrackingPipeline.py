class TrackingPipeline:
    def __init__(self, tracker):
        self.tracker = tracker

    def run(self, recognitions):
        return self.tracker.update(
            recognitions
        )