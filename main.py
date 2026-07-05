import uvicorn
from Engine.FaceEngine import FaceEngine
from Model.FaceModel import MatchResponse, FrameRequest
from Patterns import LoggerSingelton
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from Utils.Errors import NotFoundError, register_exception_handlers
from WS.ConnectionManager import manager

app = FastAPI(title="ASGARD Tracking System")
pipeline = None
register_exception_handlers(app)

@app.on_event("startup")
async def startup():
    global pipeline
    pipeline = FaceEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("WS")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    LoggerSingelton.printer("info", "🚀 Client connected successfully to /ws")

    try:
        while True:
            data = await websocket.receive_json()
            img_b64 = data.get("image_base64")
            ts = data.get("ts")
            if not img_b64:
                continue
            results = pipeline.run(img_b64)
            landmarks = []
            person_id = None

            for r in results:
                x, y, w, h = r.get("x", 0), r.get("y", 0), r.get("width", 1), r.get("height", 1)

                landmarks = [
                    {"x": x / w if w else 0.5, "y": y / h if h else 0.5, "visibility": 1}
                ]

                person_id = r.get("personName", "unknown")

            payload = {
                "ts": ts,
                "landmarks": landmarks,
                "person_id": person_id,
                "person_count": len(results)
            }

            await manager.send_personal_message(payload, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        LoggerSingelton.printer("info", "❌ Client disconnected from /ws")


@app.get("/")
async def root():
    return {"message": "Face Recognition Service Running"}


@app.post("/enroll", response_model=MatchResponse)
async def process_frame(req: FrameRequest):
    results = pipeline.run(req.image)

    print("frame received")
    if not results:
        raise NotFoundError()

    face = results[0]

    return MatchResponse(
        matched=True,
        user_id=face.get("personName", "unknown"),
        similarity=face.get("faceConfidence", 0.0)
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=False)
