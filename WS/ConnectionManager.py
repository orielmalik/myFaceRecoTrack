# WS/ConnectionManager.py
from fastapi import WebSocket
import json

class ConnectionManager:
    def __init__(self):
        # camera_id -> websocket
        self.cameras: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        self.cameras[camera_id] = websocket

    def disconnect(self, camera_id: str):
        self.cameras.pop(camera_id, None)

    async def send_to(self, camera_id: str, payload: dict):
        ws = self.cameras.get(camera_id)
        if ws:
            await ws.send_json(payload)

    async def broadcast_except(self, sender_id: str, payload: dict):
        for cam_id, ws in self.cameras.items():
            if cam_id != sender_id:
                await ws.send_json(payload)

manager = ConnectionManager()