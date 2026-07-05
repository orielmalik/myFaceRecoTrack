from pydantic import BaseModel
from typing import List
from pydantic import BaseModel

class FrameRequest(BaseModel):
    image: str
    name: str

class MatchResponse(BaseModel):
    matched: bool
    user_id: str | None
    similarity: float

class DetectionWithPersonBoundary(BaseModel):

    timestamp: str

    x: float
    y: float
    width: float
    height: float

    jointNames: List[str]
    jointXs: List[float]
    jointYs: List[float]
    jointConfidences: List[float]

    faceVector: List[float]
    faceConfidence: float

    personName: str
    personCreatedAt: str