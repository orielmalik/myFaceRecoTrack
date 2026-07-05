# Pipeline/FacePipeline.py

from Patterns.LoggerSingelton import printer

class FacePipeline:

    def __init__(self, engine, face_core, service):
        self.engine = engine
        self.core = face_core
        self.service = service

    def run(self, req):

        # 1. detect faces (MediaPipe)
        faces = self.engine.process_image(req.image)

        results = []

        for face in faces:

            # 2. embedding (InsightFace ONLY)
            embedding = self.core.get_embedding(face["crop"])

            if embedding is None:
                printer("info", "skip_no_embedding")
                continue

            # 3. identify (API)
            result = self.service.identify_face(
                embedding=embedding,
                person_name=req.name
            )

            results.append(result)

        return results