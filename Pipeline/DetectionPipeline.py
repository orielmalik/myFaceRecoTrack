class DetectionPipeline:

    def __init__(self, stages):
        self.stages = stages

    def run(self, frame):
        data = frame
        for stage in self.stages:
            data = stage.process(data)

        return data