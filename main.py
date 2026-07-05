import uvicorn
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis

from Engine.FaceCore import FaceCore
from Engine.FaceEngine import FaceEngine
from Model.BodyModel import MatchResponse, FrameRequest

from Patterns.LoggerSingelton import printer
from Pipeline.FacePipeline import FacePipeline
from Pipeline.TrackingPipeline import TrackingPipeline
from Service.FaceService import FaceService

from Utils.Errors import NotFoundError, register_exception_handlers
from WS.ConnectionManager import manager
from Utils.ApiClient import AsyncAPIClient
from Utils.Consts import *

app = FastAPI(title="ASGARD Tracking System")
engine = None
service = None
pipeline = None
tracking_pipeline = None
register_exception_handlers(app)


@app.on_event("startup")
async def startup():
    global engine, service, pipeline, face_app, face_core, tracking_pipeline
    engine = FaceEngine()

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    face_core = FaceCore(face_app)
    tracking_pipeline = TrackingPipeline(face_core )
    service = FaceService(
        AsyncAPIClient(BASE_URL_DEV),
        face_core
    )
    pipeline = FacePipeline(engine, face_core, service)
    printer("info", "System ready")


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


@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await manager.connect(websocket, camera_id)
    printer("info", f"📡 {camera_id} connected")
    try:
        while True:
            data = await websocket.receive_json()
            img_b64 = data.get("image_base64")
            ts = data.get("ts")
            if not img_b64:
                continue

            image = engine._decode_image(img_b64)
            if image is None:
                continue

            result = tracking_pipeline.run(camera_id, image)
            await self.call_api({
                "camera_id": camera_id,
                "skeleton": result["skeleton"],
                "person_id": result["person_id"]
            })
            payload = {
                "ts": ts,
                "camera_id": camera_id,
                "skeleton": result["skeleton"],
                "person_id": result["person_id"]
            }

            await manager.send_to(camera_id, payload)
            await manager.broadcast_except(camera_id, payload)

    except WebSocketDisconnect:
        manager.disconnect(camera_id)
        printer("info", f"❌ {camera_id} disconnected")


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
        name=results[0].get("personName"),
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)
