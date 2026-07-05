# Engine/PoseEngine.py
import cv2
import mediapipe as mp


class PoseEngine:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )

    def process(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        if not results.pose_landmarks:
            return None

        return results.pose_landmarks.landmark