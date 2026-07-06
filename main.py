import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis

from Engine.FaceCore import FaceCore
from Engine.FaceEngine import FaceEngine
from Engine.PoseEngine import PoseEngine
from Model.BodyModel import MatchResponse, FrameRequest

from Patterns.LoggerSingelton import printer
from Pipeline.FacePipeline import FacePipeline
from Pipeline.TrackingPipeline import TrackingPipeline
from Service.FaceService import FaceService

from Utils.Errors import NotFoundError, register_exception_handlers
from Utils.helpers import save_debug_frame
from WS.ConnectionManager import manager
from Utils.ApiClient import APIClient
from Utils.Consts import *

app = FastAPI(title="ASGARD Tracking System")
engine = None
service = None
pipeline = None
pose=None
tracking_pipeline = None
register_exception_handlers(app)


@app.on_event("startup")
async def startup():
    global engine, service, pipeline, face_app, face_core, tracking_pipeline
    engine = FaceEngine()
    pose=PoseEngine()
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    face_core = FaceCore(face_app)
    tracking_pipeline = TrackingPipeline(pose,face_core )
    service = FaceService(
        APIClient(BASE_URL_DEV),
        face_core
    )
    pipeline = FacePipeline(engine, face_core, service)
    printer("info", "System ready")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# main.py
import time
from fastapi import WebSocket, WebSocketDisconnect

latest_state: dict[str, dict] = {}

@app.websocket("../NiceGuiReco/WS/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await manager.connect(websocket, camera_id)
    printer("info", f"📡 {camera_id} connected")

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception as e:
                printer("warning", f"⚠️ bad frame from {camera_id}: {e}")
                continue

            img_b64 = data.get("image_base64")
            ts = data.get("ts")
            if not img_b64:
                continue

            try:
                image = engine._decode_image(img_b64)
                if image is None:
                    continue

                result = tracking_pipeline.run(camera_id, image)

                payload = {
                    "ts": ts,
                    "landmarks": result["skeleton"],
                    "person_id": result["person_id"],
                    "person_count": 1 if result["person_id"] != "unknown" else 0
                }

            except Exception as e:
                printer("error", f"❌ pipeline error for {camera_id}: {e}")
                continue

            latest_state[camera_id] = payload

            ok = await manager.send_to(camera_id, payload)
            if not ok:
                break

            await manager.broadcast_except(camera_id, payload)

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(camera_id)
        latest_state.pop(camera_id, None)
        printer("info", f"❌ {camera_id} disconnected")
# =========================
# ROOT
# =========================
@app.get("/")
async def root():
    return {"message": "Face Recognition Service Running"}


@app.post("/enroll", response_model=MatchResponse)
async def process_frame(req: FrameRequest):
    printer("info", "ENROLL REQUEST RECEIVED")

    t0 = time.time()
    results = pipeline.run(req)
    printer("info", f"📸 /enroll pipeline time: {time.time() - t0:.3f}s")

    if not results:
        raise NotFoundError()

    person_name = results[0].get("personName")
    embedding = results[0].get("embedding")

    image = engine._decode_image(req.image_base64)

    debug_path = save_debug_frame(
        image=image,
        landmarks=results[0].get("landmarks"),
        embedding=embedding,
        person_name=person_name or "unknown"
    )

    printer("info", f"🖼 saved debug frame: {debug_path}")

    return MatchResponse(
        name=person_name,
        embedding=embedding
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)
