from insightface.app import FaceAnalysis


class InsightFaceService:

    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0)

    def get_embedding(self, face_crop):
        faces = self.app.get(face_crop)

        if not faces:
            return None

        return faces[0].embedding