import uvicorn
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

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


# =========================
# STARTUP TIMER
# =========================
@app.on_event("startup")
async def startup():
    global engine, service, pipeline

    t0 = time.time()
    printer("info", "🚀 Starting system...")

    engine = FaceEngine()
    printer("info", f"FaceEngine loaded in {time.time() - t0:.3f}s")

    t1 = time.time()
    service = FaceService(APIClient(BASE_URL_DEV))
    printer("info", f"FaceService loaded in {time.time() - t1:.3f}s")

    t2 = time.time()
    pipeline = FacePipeline(engine, service)
    printer("info", f"FacePipeline ready in {time.time() - t2:.3f}s")

    printer("info", f"✅ TOTAL startup time: {time.time() - t0:.3f}s")


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
