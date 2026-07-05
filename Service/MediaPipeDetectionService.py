class MediaPipeDetectionService:
    def process(self, frame):
        results = self.detector.process(frame)
        faces = []

        if not results.detections:
            return faces
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            faces.append({
                "bbox": bbox,
                "frame": frame
            })

        return faces