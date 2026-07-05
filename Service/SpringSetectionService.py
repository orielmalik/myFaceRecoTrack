from Patterns import LoggerSingelton
from Utils.ApiClient import APIClient
from Utils.Consts import BASE_URL_DEV, ENDPOINT_DEV


class SpringDetectionService:

    def process(self, payload):
        client = APIClient(base_url=BASE_URL_DEV)
        response = client.post(
            ENDPOINT_DEV,
            json=
             payload

        )

        return response

LoggerSingelton.printer("info", "Starting SpringDetectionService DEMO RUN")

payload = {
    "timestamp": "2026-07-03T17:30:00",
    "x": 120.5,
    "y": 80.2,
    "width": 200.0,
    "height": 220.0,

    "jointNames": ["nose", "left_eye", "right_eye"],
    "jointXs": [120.1, 110.2, 130.3],
    "jointYs": [80.5, 75.2, 76.1],
    "jointConfidences": [0.98, 0.95, 0.96],

    "faceVector": [0.12, -0.44, 0.88, 0.31, -0.09],
    "faceConfidence": 0.93,

    "personName": "unknown",
    "personCreatedAt": "2026-07-03T17:30:00"
}


