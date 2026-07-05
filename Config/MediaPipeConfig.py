from dataclasses import dataclass


@dataclass
class MediaPipeConfig:
    static_image_mode: bool = False  # True = single image, False = video stream mode (tracking enabled)
    model_complexity: int = 1  # 0 = light/fast, 1 = balanced, 2 = heavy/accurate (Pose only)
    min_detection_confidence: float = 0.5  # minimum confidence to detect object (0-1)
    min_tracking_confidence: float = 0.5  # minimum confidence to keep tracking between frames (0-1)
    max_num_hands: int = 2  # max number of hands detected (Hands model only)
    refine_face_landmarks: bool = True  # True = better face detail (iris, lips refinement)