from Model.FaceModel import DetectionWithPersonBoundary
from Patterns.LoggerSingelton import printer
from Utils.Consts import ENDPOINT_DEV


class FaceService:

    def __init__(self, api_client):
        self.api = api_client

    def process_face(self, face_data):
        embedding = face_data["embedding"]
        person_name = face_data.get("person_name")

        dto = DetectionWithPersonBoundary(
            personName=person_name,
            faceVector=embedding.tolist() if hasattr(embedding, "tolist") else embedding
        )

        response = self.api.post(
            ENDPOINT_DEV,
            json=dto.model_dump(exclude_none=True, exclude_defaults=True)
        )

        return {
            "status": response["matched"],
            "personName": response["personName"],
        }
