# Engine/FaceCore.py
class FaceCore:
    def __init__(self, face_app):
        self.face_app = face_app

    def get_embedding(self, image_bgr):
        if image_bgr is None:
            return None
        faces = self.face_app.get(image_bgr)
        if not faces:
            return None
        return faces[0].embedding