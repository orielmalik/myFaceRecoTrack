from pydantic import BaseModel
from typing import List
from pydantic import BaseModel


class FrameRequest(BaseModel):
    image: str
    name: str


class MatchResponse(BaseModel):
    matched: bool
    name: str


from typing import List, Optional
from pydantic import BaseModel, Field


class DetectionWithPersonBoundary(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    matched: bool = False
    jointNames: List[str] = Field(default_factory=list)
    jointXs: List[float] = Field(default_factory=list)
    jointYs: List[float] = Field(default_factory=list)
    jointConfidences: List[float] = Field(default_factory=list)
    personName: Optional[str] = None
    faceVector: List[float] = Field(default_factory=list)
    faceConfidence: float = 0.0



class PoseLandmark(BaseModel):
    name: Optional[str] = None
    x: float
    y: float
    confidence: float


class DetectionUpdateRequest(BaseModel):
    cameraId: str
    ts: Optional[str] = None

    faceVector: List[float] = Field(default_factory=list)
    faceConfidence: float = 0.0

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    skeleton: List[PoseLandmark] = Field(default_factory=list)


class DetectionResponse(BaseModel):
    matched: bool = False

    personName: Optional[str] = None

    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None

    faceConfidence: float = 0.0
    faceVector: List[float] = Field(default_factory=list)

    jointNames: List[str] = Field(default_factory=list)
    jointXs: List[float] = Field(default_factory=list)
    jointYs: List[float] = Field(default_factory=list)
    jointConfidences: List[float] = Field(default_factory=list)
