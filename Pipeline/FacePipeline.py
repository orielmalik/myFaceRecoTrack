from Patterns.LoggerSingelton import printer

class FacePipeline:

    def __init__(self, engine, service):
        self.engine = engine
        self.service = service

    def run(self, req):
        faces = self.engine.process_image(req.image)
        results = []
        for face in faces:
            embedding = self.service.get_embedding(face["crop"])

            if embedding is None:
                printer("info", "skip_no_embedding")
                continue

            result = self.service.identify_face(
                embedding=embedding,
                person_name=req.name
            )

            results.append(result)

        return results