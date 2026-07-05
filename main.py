import uvicorn
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis

from Engine.FaceEngine import FaceEngine
from Model.FaceModel import MatchResponse, FrameRequest

from Patterns.LoggerSingelton import printer
from Pipeline.FacePipeline import FacePipeline
from Service.FaceService import FaceService

from Utils.Errors import NotFoundError, register_exception_handlers
from WS.ConnectionManager import manager
from Utils.ApiClient import APIClient
from Utils.Consts import *

app = FastAPI(title="ASGARD Tracking System")
engine = None
service = None
pipeline = None

register_exception_handlers(app)


@app.on_event("startup")
async def startup():
    global engine, service, pipeline, face_app

    printer("info", "🚀 Starting system...")

    # 1. CV engine (MediaPipe)
    engine = FaceEngine()

    # 2. InsightFace (identity model)
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    service = FaceService(
        APIClient(BASE_URL_DEV),
        face_app
    )
    pipeline = FacePipeline(engine, service)

    printer("info", "✅ System ready")


# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# ROOT
# =========================
@app.get("/")
async def root():
    return {"message": "Face Recognition Service Running"}


# =========================
# ENROLL
# =========================
@app.post("/enroll", response_model=MatchResponse)
async def process_frame(req: FrameRequest):
    printer("info", "ENROLL REQUEST RECEIVED")

    t0 = time.time()

    results = pipeline.run(req)

    printer("info", f"📸 /enroll pipeline time: {time.time() - t0:.3f}s")

    if not results:
        raise NotFoundError()

    return MatchResponse(
        matched=results[0].get("matched"),
        user_id=results[0].get("personName"),
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)
