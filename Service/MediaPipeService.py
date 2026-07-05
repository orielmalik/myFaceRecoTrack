import cv2
import numpy as np
import base64
import mediapipe as mp


class MediaPipeService:

    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

    def extract(self, image_base64: str):

        img_bytes = base64.b64decode(image_base64)
        np_img = np.frombuffer(img_bytes, np.uint8)

        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.detector.process(rgb)

        if not results.detections:
            return {"face_detected": False, "faces": []}

        h, w, _ = image.shape

        faces = []

        for det in results.detections:

            bbox = det.location_data.relative_bounding_box

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            crop = image[y1:y2, x1:x2]

            faces.append({
                "bbox": [x1, y1, x2, y2],
                "crop": crop
            })

        return {
            "face_detected": True,
            "faces": faces
        }