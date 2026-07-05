import base64
import cv2
import numpy as np
import mediapipe as mp
from insightface.app import FaceAnalysis
from Patterns.LoggerSingelton import printer


class FaceEngine:

    def __init__(self):

        # MediaPipe detector (only for bbox)
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

        # InsightFace (embedding)
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def _is_valid_face(self, det_score, x1, y1, x2, y2, w, h):
        # 1. detection confidence
        confidence_score = float(det_score)

        # 2. face size ratio (scale independent)
        face_area = (x2 - x1) * (y2 - y1)
        image_area = w * h
        area_ratio = face_area / image_area

        # 3. aspect ratio sanity check (avoid weird crops)
        aspect_ratio = (x2 - x1) / ((y2 - y1) + 1e-6)

        # scoring logic
        if confidence_score < 0.5:
            return False, "low_confidence"

        if area_ratio < 0.003:
            return False, "face_too_small"

        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False, "invalid_shape"

        return True, "ok"
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
                printer("error", f"Base64 decode failed: {e}")
                return None

        if image is None or not hasattr(image, "shape"):
            return None

        return image

    # -------------------------
    # FACE DETECTION (MediaPipe)
    # -------------------------
    def extract_faces(self, image):
        printer("info", "Running face detection")

        image = self._decode_image(image)
        if image is None:
            printer("error", "Invalid image format")
            return []

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            printer("warning", "No faces found (MediaPipe)")
            return []

        h, w, _ = image.shape
        faces = []

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))

            is_valid, reason = self._is_valid_face(
                det.score[0], x1, y1, x2, y2, w, h
            )

            if not is_valid:
                printer("warning", f"Face rejected: {reason}")
                continue

            crop = image[y1:y2, x1:x2]
            crop = cv2.resize(crop, (640, 640))

            faces.append({
                "bbox": (x1, y1, x2, y2),
                "crop": crop,
                "confidence": float(det.score[0]),
                "quality": reason
            })
        printer("info", f"Detected {len(faces)} faces")
        return faces

    # -------------------------
    # EMBEDDING (InsightFace)
    # -------------------------
    def get_embedding(self, crop):
        if crop is None:
            return None

        try:
            # Ensure correct format (InsightFace prefers BGR)
            if crop.shape[-1] != 3:
                printer("warning", "Invalid crop format")
                return None

            faces = self.app.get(crop)

            if not faces or len(faces) == 0:
                printer("warning", "No faces detected by InsightFace")
                return None

            embedding = faces[0].embedding

            if embedding is None or len(embedding) == 0:
                return None

            return embedding

        except Exception as e:
            printer("error", f"Embedding failure: {e}")
            return None