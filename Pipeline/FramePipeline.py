class FramePipeline:

    def __init__(
        self,
        detection_pipeline,
        recognition_pipeline,
        tracking_pipeline
    ):
        self.detection_pipeline = detection_pipeline
        self.recognition_pipeline = recognition_pipeline
        self.tracking_pipeline = tracking_pipeline

    def process(self, frame):

        detections = self.detection_pipeline.run(frame)

        recognitions = []

        for detection in detections:

            result = self.recognition_pipeline.run(
                detection
            )

            if result:
                recognitions.append(result)

        return self.tracking_pipeline.run(
            recognitions
        )