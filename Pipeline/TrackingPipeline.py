from Engine.PoseEngine import PoseEngine


class TrackingPipeline:
    def __init__(self, face_core, recognize_every=15):
        self.pose_engine = PoseEngine()
        self.face_core = face_core
        self.recognize_every = recognize_every
        self.frame_counters = {}      # camera_id -> counter
        self.last_identity = {}       # camera_id -> person_id

    def run(self, camera_id, image_bgr):
        skeleton = self.pose_engine.process(image_bgr)

        count = self.frame_counters.get(camera_id, 0)
        person_id = self.last_identity.get(camera_id, "unknown")

        if count % self.recognize_every == 0:
            embedding = self.face_core.get_embedding(image_bgr)
            if embedding is not None:
                match = self.matcher.find_best_match(embedding)
                person_id = match["user_id"] if match["matched"] else "unknown"
                self.last_identity[camera_id] = person_id

        self.frame_counters[camera_id] = count + 1

        return {
            "skeleton": skeleton,
            "person_id": person_id
        }