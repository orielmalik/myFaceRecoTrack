class RecognitionPipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self, face):
        data = face
        for stage in self.stages:
            data = stage.process(data)
        return data