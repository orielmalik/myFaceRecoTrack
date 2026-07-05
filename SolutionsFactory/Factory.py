from Utils.helpers import get_data_from_json
import mediapipe as mp
class MediaPipeSolutionFactory:
    def __init__(self, json_file="MediaPipeTypes.json"):
        self.mapping = get_data_from_json(json_file)[1]

    def create_module(self, solution_key: str):
        key = solution_key.upper()
        if key not in self.mapping:
            raise ValueError(f"Unknown solution: {solution_key}")
        module_name = self.mapping[key]
        return getattr(mp.solutions, module_name)
