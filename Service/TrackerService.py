class TrackerService:

    def __init__(self):
        self.tracks = {}

    def update(self, recognitions):

        for recognition in recognitions:

            person_id = recognition["personId"]

            self.tracks[person_id] = recognition

        return self.tracks