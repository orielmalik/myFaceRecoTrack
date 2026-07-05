from Config.MediaPipeConfig import MediaPipeConfig
class MediaPipeConfigBuilder:

    def __init__(self):
        self.config = MediaPipeConfig()

    def detection_confidence(self, value):
        self.config.min_detection_confidence = value
        return self

    def tracking_confidence(self, value):
        self.config.min_tracking_confidence = value
        return self

    def model_complexity(self, value):
        self.config.model_complexity = value
        return self

    def build(self):
        return self.config
