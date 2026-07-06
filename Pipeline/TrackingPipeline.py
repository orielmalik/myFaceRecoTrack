from datetime import datetime
from Utils.ApiClient import APIClient
from Utils.Consts import BASE_URL_DEV, ENDPOINT_TrackDEV
from Utils.helpers import convert_skeleton_to_dto


class TrackingPipeline:
    def __init__(self, pose, face_core, recognize_every=15):
        self.pose_engine = pose
        self.face_core = face_core
        self.apiClient = APIClient(BASE_URL_DEV)
        self.recognize_every = recognize_every
        self.frame_counters = {}
        self.last_identity = {}

    def run(self, camera_id, image_bgr):

        skeleton = self.pose_engine.process(image_bgr)
        count = self.frame_counters.get(camera_id, 0)
        person_id = self.last_identity.get(camera_id, "unknown")

        if count % self.recognize_every == 0:

            embedding = self.face_core.get_embedding(image_bgr)

            if embedding is not None:

                payload = build_tracking_payload(
                    camera_id,
                    embedding,
                    skeleton
                )

                match = self.apiClient.put(ENDPOINT_TrackDEV, payload)

                if match and isinstance(match, dict):
                    if match.get("matched") is not False:
                        person_id = match.get("personName", "unknown")
                        self.last_identity[camera_id] = person_id

        self.frame_counters[camera_id] = count + 1

        return {
            "skeleton": skeleton or [],
            "person_id": person_id
        }


def build_tracking_payload(camera_id, embedding, skeleton):
    return {
        "cameraId": camera_id,
        "ts": datetime.utcnow().isoformat(),

        "faceVector": embedding.tolist() if hasattr(embedding, "tolist") else embedding,

        "faceConfidence": 0.9,

        "x": 0.0,
        "y": 0.0,
        "width": 0.0,
        "height": 0.0,

        "skeleton": convert_skeleton_to_dto(skeleton)
    }
