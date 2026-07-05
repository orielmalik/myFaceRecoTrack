class CropFaceService:

    def process(self, detections):
        faces = []
        for detection in detections:

            crop = self.extract(
                detection["frame"],
                detection["bbox"]
            )

            faces.append(crop)

        return faces