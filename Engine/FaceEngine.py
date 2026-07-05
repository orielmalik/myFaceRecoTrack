import cv2
import numpy as np
import base64

class FaceEngine:

    def __init__(self):
        import mediapipe as mp
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    # -------------------------
    # DECODE BASE64 → IMAGE
    # -------------------------
    def _decode_image(self, image):

        if isinstance(image, str):

            try:
                # אם זה data URL
                if "," in image:
                    image = image.split(",")[1]

                img_bytes = base64.b64decode(image)
                nparr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            except Exception as e:
                print("decode failed:", e)
                return None

        return image

    # -------------------------
    # MAIN PIPELINE
    # -------------------------
    def process_image(self, image):

        # 🔥 חובה decode ראשון
        image = self._decode_image(image)

        if image is None:
            return []

        if not isinstance(image, np.ndarray):
            return []

        # עכשיו זה בטוח
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
                "crop": crop,
                "bbox": (x1, y1, x2, y2)
            })

        return faces