import cv2
import numpy as np
import base64
import mediapipe as mp
from Patterns.LoggerSingelton import printer


class FaceEngine:

    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    # -------------------------
    # IMAGE DECODING
    # -------------------------
    def _decode_image(self, image):
        if isinstance(image, str):
            try:
                if "," in image:
                    image = image.split(",")[1]

                img_bytes = base64.b64decode(image)
                nparr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            except Exception as e:
                printer("error", f"decode failed: {e}")
                return None

        if image is None or not hasattr(image, "shape"):
            return None

        return image

    # -------------------------
    # FACE DETECTION ONLY
    # -------------------------
    def process_image(self, image):
        image = self._decode_image(image)
        if image is None:
            return []

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return []

        h, w, _ = image.shape
        faces = []

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))

            crop = image[y1:y2, x1:x2]

            faces.append({
                "bbox": (x1, y1, x2, y2),
                "crop": crop,
                "confidence": float(det.score[0])
            })

        return faces