# Service/FaceService.py

from Model.BodyModel import DetectionWithPersonBoundary
from Utils.Consts import ENDPOINT_DEV



class FaceService:

    def __init__(self, api_client, face_core):
        self.api = api_client
        self.core = face_core

    def identify_face(self, embedding, person_name=None):

        dto = DetectionWithPersonBoundary(
            personName=person_name,
            faceVector=embedding.tolist()
        )

        return self.api.post(
            ENDPOINT_DEV,
            json=dto.model_dump()
        )
