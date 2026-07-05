import cv2
import numpy as np
import base64
import mediapipe as mp
from insightface.app import FaceAnalysis

from Utils.ApiClient import APIClient


class FaceEngine:

    def __init__(self):

        # MediaPipe detector
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        # InsightFace
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.api = APIClient(base_url="http://localhost:8088")

    def run(self, image_base64: str):

        img_bytes = base64.b64decode(image_base64)
        np_img = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.detector.process(rgb)

        if not results.detections:
            return {"faces": []}

        h, w, _ = image.shape
        responses = []

        for det in results.detections:

            if det.score[0] < 0.6:
                continue

            bbox = det.location_data.relative_bounding_box

            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            faces = self.app.get(crop)
            if not faces:
                continue

            embedding = faces[0].embedding.tolist()

            try:
                response = self.api.post(
                    "/cross/detections",
                    json={
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "faceVector": embedding,
                        "faceConfidence": float(det.score[0]),
                        "jointNames": [],
                        "jointXs": [],
                        "jointYs": [],
                        "jointConfidences": [],
                        "personName": "unknown"
                    }
                )

                responses.append(response)

            except Exception as e:
                responses.append({"error": str(e)})

        return responses

    def close(self):
        self.api.close()