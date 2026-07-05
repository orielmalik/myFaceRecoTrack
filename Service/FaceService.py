from Model.FaceModel import DetectionWithPersonBoundary
from Utils.Consts import ENDPOINT_DEV


class FaceService:

    def __init__(self, api_client, face_app):
        self.api = api_client
        self.face_app = face_app

    # -------------------------
    # EMBEDDING (InsightFace)
    # -------------------------
    def get_embedding(self, crop):
        if crop is None:
            return None

        try:
            faces = self.face_app.get(crop)

            if not faces:
                return None

            return faces[0].embedding

        except Exception:
            return None

    # -------------------------
    # IDENTITY (Spring Boot)
    # -------------------------
    def identify_face(self, embedding, person_name=None):

        dto = DetectionWithPersonBoundary(
            personName=person_name,
            faceVector=embedding.tolist()
        )

        response = self.api.post(
            ENDPOINT_DEV,
            json=dto.model_dump(exclude_none=True, exclude_defaults=True)
        )

        return {
            "status": response["matched"],
            "personName": response["personName"],
        }