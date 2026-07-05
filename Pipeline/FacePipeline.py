from Patterns.LoggerSingelton import printer


class FacePipeline:

    def __init__(self, engine, service):
        self.engine = engine
        self.service = service

    def run(self, req):
        faces = self.engine.extract_faces(req.image)

        results = []

        for face in faces:
            embedding = self.engine.get_embedding(face["crop"])
            if not embedding:
                printer("info", "continiueNotEmbed")
                continue

            result = self.service.process_face({
                "embedding": embedding,
                "confidence": face["confidence"],
                "bbox": face["bbox"],
                "person_name": req.name
            })
            printer("info", "results.append")

            results.append(result)

        return results