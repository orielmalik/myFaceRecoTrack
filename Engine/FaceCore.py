# Engine/FaceCore.py
import numpy as np

class FaceCore:

    def __init__(self, face_app):
        self.face_app = face_app

    def get_embedding(self, image):
        """
        ONLY InsightFace -> returns vector
        """

        if image is None:
            return None

        # InsightFace expects RGB
        if hasattr(image, "shape"):
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = self.face_app.get(image)

        if not faces:
            return None

        return faces[0].embedding