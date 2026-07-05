# Pipeline/FacePipeline.py

from Patterns.LoggerSingelton import printer

class FacePipeline:

    def __init__(self, engine, face_core, service):
        self.engine = engine
        self.core = face_core
        self.service = service

    # Pipeline/FacePipeline.py
    def run(self, req):
        image = self.engine._decode_image(req.image)
        embedding = self.core.get_embedding(image)
        if embedding is None:
            printer("info", "skip_no_embedding")
            return []
        return [self.service.identify_face(embedding=embedding, person_name=req.name)]